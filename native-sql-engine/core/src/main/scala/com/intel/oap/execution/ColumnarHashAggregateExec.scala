/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.oap.execution

import com.intel.oap.ColumnarPluginConfig
import com.intel.oap.expression._
import com.intel.oap.vectorized._
import com.google.common.collect.Lists
import java.util.concurrent.TimeUnit._

import org.apache.arrow.gandiva.expression._
import org.apache.arrow.gandiva.evaluator._
import org.apache.arrow.vector.types.pojo.ArrowType
import org.apache.arrow.vector.types.pojo.Field
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.spark.TaskContext
import org.apache.spark.memory.{SparkOutOfMemoryError, TaskMemoryManager}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.{UserAddedJarUtils, Utils, ExecutorManager}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.errors._
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.BindReferences.bindReferences
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.codegen.Block._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.catalyst.util.DateTimeUtils._
import org.apache.spark.sql.catalyst.util.truncatedString
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.aggregate._
import org.apache.spark.sql.execution.datasources.v2.arrow.SparkMemoryUtils
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.vectorized.MutableColumnarRow
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.KVIterator
import scala.collection.JavaConverters._

import scala.collection.Iterator
import scala.util.control.Breaks._

/**
 * Columnar Based HashAggregateExec.
 */
case class ColumnarHashAggregateExec(
    requiredChildDistributionExpressions: Option[Seq[Expression]],
    groupingExpressions: Seq[NamedExpression],
    aggregateExpressions: Seq[AggregateExpression],
    aggregateAttributes: Seq[Attribute],
    initialInputBufferOffset: Int,
    resultExpressions: Seq[NamedExpression],
    child: SparkPlan)
    extends BaseAggregateExec
    with ColumnarCodegenSupport
    with AliasAwareOutputPartitioning {

  val sparkConf = sparkContext.getConf
  val numaBindingInfo = ColumnarPluginConfig.getConf.numaBindingInfo
  override def supportsColumnar = true

  // Members declared in org.apache.spark.sql.execution.AliasAwareOutputPartitioning
  override protected def outputExpressions: Seq[NamedExpression] = resultExpressions

  // Members declared in org.apache.spark.sql.execution.CodegenSupport
  protected def doProduce(ctx: CodegenContext): String = throw new UnsupportedOperationException()

  // Members declared in org.apache.spark.sql.catalyst.plans.QueryPlan
  override def output: Seq[Attribute] = resultExpressions.map(_.toAttribute)

  // Members declared in org.apache.spark.sql.execution.SparkPlan
  protected override def doExecute()
      : org.apache.spark.rdd.RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new UnsupportedOperationException()

  override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "output_batches"),
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "input_batches"),
    "aggTime" -> SQLMetrics.createTimingMetric(sparkContext, "time in aggregation process"),
    "processTime" -> SQLMetrics.createTimingMetric(sparkContext, "totaltime_hashagg"))

  val numOutputRows = longMetric("numOutputRows")
  val numOutputBatches = longMetric("numOutputBatches")
  val numInputBatches = longMetric("numInputBatches")
  val aggTime = longMetric("aggTime")
  val totalTime = longMetric("processTime")
  numOutputRows.set(0)
  numOutputBatches.set(0)
  numInputBatches.set(0)

  buildCheck()

  val onlyResultExpressions: Boolean =
    if (groupingExpressions.isEmpty && aggregateExpressions.isEmpty &&
      child.output.isEmpty && resultExpressions.nonEmpty) true
    else false

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    var eval_elapse: Long = 0
    child.executeColumnar().mapPartitions { iter =>
      ExecutorManager.tryTaskSet(numaBindingInfo)
      val native_function = TreeBuilder.makeFunction(
        s"standalone",
        Lists.newArrayList(getKernelFunction),
        new ArrowType.Int(32, true))
      val hash_aggr_expr =
        TreeBuilder
          .makeExpression(native_function, Field.nullable("result", new ArrowType.Int(32, true)))
      val hash_aggr_input_schema = ConverterUtils.toArrowSchema(child.output)
      val hash_aggr_out_schema = ConverterUtils.toArrowSchema(output)
      val resultStructType = ArrowUtils.fromArrowSchema(hash_aggr_out_schema)
      val nativeKernel = new ExpressionEvaluator()
      nativeKernel
        .build(
          hash_aggr_input_schema,
          Lists.newArrayList(hash_aggr_expr),
          hash_aggr_out_schema,
          true)
      val nativeIterator = nativeKernel.finishByIterator()

      def close = {
        aggTime += (eval_elapse / 1000000)
        totalTime += (eval_elapse / 1000000)
        nativeKernel.close
        nativeIterator.close
      }

      var numRowsInput = 0
      var hasNextCount = 0
      // now we can return this wholestagecodegen iter
      val res = new Iterator[ColumnarBatch] {
        var processed = false
        /** Special cases need to be handled in scala side:
         * (1) aggregate literal (2) only result expressions
         * (3) empty input (4) grouping literal
         */
        var skip_count = false
        var skip_native = false
        var onlyResExpr = false
        var emptyInput = false
        var count_num_row = 0
        def process: Unit = {
          while (iter.hasNext) {
            val cb = iter.next()
            numInputBatches += 1
            if (cb.numRows != 0) {
              numRowsInput += cb.numRows
              val beforeEval = System.nanoTime()
              if (hash_aggr_input_schema.getFields.size != 0) {
                val input_rb =
                  ConverterUtils.createArrowRecordBatch(cb)
                nativeIterator.processAndCacheOne(hash_aggr_input_schema, input_rb)
                ConverterUtils.releaseArrowRecordBatch(input_rb)
              } else {
                // Special case for no input batch
                if ((aggregateExpressions.nonEmpty && aggregateExpressions.head
                     .aggregateFunction.children.head.isInstanceOf[Literal]) ||
                    (groupingExpressions.nonEmpty && groupingExpressions.head.children.nonEmpty &&
                     groupingExpressions.head.children.head.isInstanceOf[Literal])) {
                  // This is a special case used by literal aggregation
                  skip_native = true
                  breakable{
                    for (exp <- aggregateExpressions) {
                      if (exp.aggregateFunction.isInstanceOf[Count]) {
                        skip_count = true
                        count_num_row += cb.numRows
                        break
                      }
                    }
                  }
                }
              }
              eval_elapse += System.nanoTime() - beforeEval
            }
          }
          processed = true
        }
        override def hasNext: Boolean = {
          hasNextCount += 1
          if (!processed) process
          if (skip_count) {
            count_num_row > 0
          } else if (skip_native) {
            hasNextCount == 1
          } else if (onlyResultExpressions && hasNextCount == 1) {
            onlyResExpr = true
            true
          } else if (!onlyResultExpressions && groupingExpressions.isEmpty &&
                     numRowsInput == 0 && hasNextCount == 1) {
            emptyInput = true
            true
          } else {
            nativeIterator.hasNext
          }
        }

        override def next(): ColumnarBatch = {
          if (!processed) process
          val beforeEval = System.nanoTime()
          if (skip_native) {
            // special handling for literal aggregation and grouping
            getResForAggregateAndGroupingLiteral
          } else if (onlyResExpr) {
            // special handling for only result expressions
            getResForOnlyResExpr
          } else if (emptyInput) {
            // special handling for empty input batch
            getResForEmptyInput
          } else {
            val output_rb = nativeIterator.next
            if (output_rb == null) {
              eval_elapse += System.nanoTime() - beforeEval
              val resultColumnVectors =
                ArrowWritableColumnVector.allocateColumns(0, resultStructType)
              return new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
            }
            val outputNumRows = output_rb.getLength
            val output = ConverterUtils.fromArrowRecordBatch(hash_aggr_out_schema, output_rb)
            ConverterUtils.releaseArrowRecordBatch(output_rb)
            eval_elapse += System.nanoTime() - beforeEval
            numOutputRows += outputNumRows
            numOutputBatches += 1
            new ColumnarBatch(output.map(v => v.asInstanceOf[ColumnVector]), outputNumRows)
          }
        }
        def putDataIntoVector(vectors: Array[ArrowWritableColumnVector],
                              res: Any, idx: Int): Unit = {
          if (res == null) {
            vectors(idx).putNull(0)
          } else {
            vectors(idx).dataType match {
              case t: IntegerType =>
                vectors(idx)
                  .put(0, res.asInstanceOf[Number].intValue)
              case t: LongType =>
                vectors(idx)
                  .put(0, res.asInstanceOf[Number].longValue)
              case t: DoubleType =>
                vectors(idx)
                  .put(0, res.asInstanceOf[Number].doubleValue())
              case t: FloatType =>
                vectors(idx)
                  .put(0, res.asInstanceOf[Number].floatValue())
              case t: ByteType =>
                vectors(idx)
                  .put(0, res.asInstanceOf[Number].byteValue())
              case t: ShortType =>
                vectors(idx)
                  .put(0, res.asInstanceOf[Number].shortValue())
              case t: StringType =>
                val values = (res :: Nil).map(_.toString).map(_.toByte).toArray
                vectors(idx).putBytes(0, 1, values, 0)
              case t: BooleanType =>
                vectors(idx)
                  .put(0, res.asInstanceOf[Boolean].booleanValue())
              case other =>
                throw new UnsupportedOperationException(s"$other is not supported.")
            }
          }
        }
        def getResForAggregateAndGroupingLiteral: ColumnarBatch = {
          val resultColumnVectors =
            ArrowWritableColumnVector.allocateColumns(1, resultStructType)
          for (vector <- resultColumnVectors) {
            vector.getValueVector.setValueCount(1)
          }
          var idx = 0
          for (exp <- groupingExpressions) {
            val out_res = exp.children.head.asInstanceOf[Literal].value
            putDataIntoVector(resultColumnVectors, out_res, idx)
            idx += 1
          }
          for (exp <- aggregateExpressions) {
            val mode = exp.mode
            val aggregateFunc = exp.aggregateFunction
            val out_res = aggregateFunc.children.head.asInstanceOf[Literal].value
            aggregateFunc match {
              case Sum(_) =>
                mode match {
                  case Partial | PartialMerge =>
                    val sum = aggregateFunc.asInstanceOf[Sum]
                    val aggBufferAttr = sum.inputAggBufferAttributes
                    // decimal sum check sum.resultType
                    if (aggBufferAttr.size == 2) {
                      putDataIntoVector(resultColumnVectors, out_res, idx) // sum
                      idx += 1
                      putDataIntoVector(resultColumnVectors, false, idx) // isEmpty
                      idx += 1
                    } else {
                      putDataIntoVector(resultColumnVectors, out_res, idx)
                      idx += 1
                    }
                  case Final =>
                    putDataIntoVector(resultColumnVectors, out_res, idx)
                    idx += 1
                }
              case Average(_) =>
                mode match {
                  case Partial | PartialMerge =>
                    putDataIntoVector(resultColumnVectors, out_res, idx) // sum
                    idx += 1
                    if (out_res == null) {
                      putDataIntoVector(resultColumnVectors, 0, idx) // count
                    } else {
                      putDataIntoVector(resultColumnVectors, 1, idx) // count
                    }
                    idx += 1
                  case Final =>
                    putDataIntoVector(resultColumnVectors, out_res, idx)
                    idx += 1
                }
              case Count(_) =>
                putDataIntoVector(resultColumnVectors, count_num_row, idx)
                idx += 1
              case Max(_) | Min(_) =>
                putDataIntoVector(resultColumnVectors, out_res, idx)
                idx += 1
              case StddevSamp(_, _) =>
                mode match {
                  case Partial =>
                    putDataIntoVector(resultColumnVectors, 1, idx) // n
                    idx += 1
                    putDataIntoVector(resultColumnVectors, out_res, idx) // avg
                    idx += 1
                    putDataIntoVector(resultColumnVectors, 0, idx) // m2
                    idx += 1
                  case Final =>
                    if (aggregateFunc.asInstanceOf[StddevSamp].nullOnDivideByZero) {
                      putDataIntoVector(resultColumnVectors, null, idx)
                    } else {
                      putDataIntoVector(resultColumnVectors, Double.NaN, idx)
                    }
                    idx += 1
                }
            }
          }
          count_num_row = 0
          new ColumnarBatch(
            resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 1)
        }
        def getResForOnlyResExpr: ColumnarBatch = {
          // This function has limited support for only-result-expression case.
          // Fake input for projection:
          val inputColumnVectors =
            ArrowWritableColumnVector.allocateColumns(0, resultStructType)
          val valueVectors =
            inputColumnVectors.map(columnVector => columnVector.getValueVector).toList
          val projector = ColumnarProjection.create(child.output, resultExpressions)
          val resultColumnVectorList = projector.evaluate(1, valueVectors)
          new ColumnarBatch(
            resultColumnVectorList.map(v => v.asInstanceOf[ColumnVector]).toArray,
            1)
        }
        def getResForEmptyInput: ColumnarBatch = {
          if (aggregateExpressions.isEmpty) {
            // To align with spark, in this case, one empty row is returned.
            val resultColumnVectors =
              ArrowWritableColumnVector.allocateColumns(0, resultStructType)
            return new ColumnarBatch(
              resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 1)
          }
          val resultColumnVectors =
            ArrowWritableColumnVector.allocateColumns(1, resultStructType)
          for (vector <- resultColumnVectors) {
            vector.getValueVector.setValueCount(1)
          }
          // If groupby is not required, for Final mode, a default value will be
          // returned if input is empty.
          var idx = 0
          for (expr <- aggregateExpressions) {
            expr.aggregateFunction match {
              case Average(_) | StddevSamp(_, _) | Sum(_) | Max(_) | Min(_) =>
                expr.mode match {
                  case Final =>
                    resultColumnVectors(idx).putNull(0)
                    idx += 1
                  case _ =>
                }
              case Count(_) =>
                expr.mode match {
                  case Final =>
                    val out_res = 0
                    resultColumnVectors(idx).dataType match {
                      case t: IntegerType =>
                        resultColumnVectors(idx)
                          .put(0, out_res.asInstanceOf[Number].intValue)
                      case t: LongType =>
                        resultColumnVectors(idx)
                          .put(0, out_res.asInstanceOf[Number].longValue)
                      case t: DoubleType =>
                        resultColumnVectors(idx)
                          .put(0, out_res.asInstanceOf[Number].doubleValue())
                      case t: FloatType =>
                        resultColumnVectors(idx)
                          .put(0, out_res.asInstanceOf[Number].floatValue())
                      case t: ByteType =>
                        resultColumnVectors(idx)
                          .put(0, out_res.asInstanceOf[Number].byteValue())
                      case t: ShortType =>
                        resultColumnVectors(idx)
                          .put(0, out_res.asInstanceOf[Number].shortValue())
                      case t: StringType =>
                        val values = (out_res :: Nil).map(_.toByte).toArray
                        resultColumnVectors(idx)
                          .putBytes(0, 1, values, 0)
                    }
                    idx += 1
                  case _ =>
                }
              case other =>
                throw new UnsupportedOperationException(s"not currently supported: $other.")
            }
          }
          // will only put default value for Final mode
          aggregateExpressions.head.mode match {
            case Final =>
              new ColumnarBatch(
                resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 1)
            case _ =>
              new ColumnarBatch(
                resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
          }
        }
      }
      SparkMemoryUtils.addLeakSafeTaskCompletionListener[Unit](_ => {
        close
      })
      new CloseableColumnBatchIterator(res)
    }
  }

  def buildCheck(): Unit = {
    // check input datatype
    for (attr <- child.output) {
      try {
        ConverterUtils.checkIfTypeSupported(attr.dataType)
      } catch {
        case e: UnsupportedOperationException =>
          throw new UnsupportedOperationException(
            s"${attr.dataType} is not supported in ColumnarAggregation")
      }
    }
    // check output datatype
    resultExpressions.foreach(expr => {
      try {
        ConverterUtils.checkIfTypeSupported(expr.dataType)
      } catch {
        case e: UnsupportedOperationException =>
          throw new UnsupportedOperationException(
            s"${expr.dataType} is not supported in ColumnarAggregation")
      }
    })
    // check project
    for (expr <- aggregateExpressions) {
      val internalExpressionList = expr.aggregateFunction.children
      ColumnarProjection.buildCheck(child.output, internalExpressionList)
    }
    ColumnarProjection.buildCheck(child.output, groupingExpressions)
    ColumnarProjection.buildCheck(child.output, resultExpressions)
    // check the supported types and modes for different aggregate functions
    checkTypeAndAggrFunction(aggregateExpressions, aggregateAttributes)
  }

  // This method checks the supported types and modes for different aggregate functions.
  def checkTypeAndAggrFunction(aggregateExpressions: Seq[AggregateExpression],
                               aggregateAttributeList: Seq[Attribute]): Unit = {
    var res_index = 0
    for (expIdx <- aggregateExpressions.indices) {
      val exp: AggregateExpression = aggregateExpressions(expIdx)
      if (exp.filter.isDefined) {
        throw new UnsupportedOperationException(
          "filter is not supported in AggregateExpression")
      }
      val mode = exp.mode
      val aggregateFunc = exp.aggregateFunction
      aggregateFunc match {
        case Average(_) =>
          val supportedTypes = List(ByteType, ShortType, IntegerType, LongType,
            FloatType, DoubleType, DateType, BooleanType)
          val avg = aggregateFunc.asInstanceOf[Average]
          val aggBufferAttr = avg.inputAggBufferAttributes
          for (index <- aggBufferAttr.indices) {
            val attr = ConverterUtils.getAttrFromExpr(aggBufferAttr(index))
            if (supportedTypes.indexOf(attr.dataType) == -1 &&
                !attr.dataType.isInstanceOf[DecimalType]) {
              throw new UnsupportedOperationException(
                s"${attr.dataType} is not supported in Columnar Average")
            }
          }
          mode match {
            case Partial =>
              res_index += 2
            case PartialMerge => res_index += 1
            case Final => res_index += 1
            case other =>
              throw new UnsupportedOperationException(
                s"${other} is not supported in Columnar Average")
          }
        case Sum(_) =>
          val supportedTypes = List(ByteType, ShortType, IntegerType, LongType,
            FloatType, DoubleType, DateType, BooleanType)
          val sum = aggregateFunc.asInstanceOf[Sum]
          val aggBufferAttr = sum.inputAggBufferAttributes
          val attr = ConverterUtils.getAttrFromExpr(aggBufferAttr.head)
          if (supportedTypes.indexOf(attr.dataType) == -1 &&
              !attr.dataType.isInstanceOf[DecimalType]) {
            throw new UnsupportedOperationException(
              s"${attr.dataType} is not supported in Columnar Sum")
          }
          mode match {
            case Partial | PartialMerge =>
              if (aggBufferAttr.size == 2) {
                // decimal sum check sum.resultType
                res_index += 2
              } else {
                res_index += 1
              }
            case Final => res_index += 1
            case other =>
              throw new UnsupportedOperationException(
                s"${other} is not supported in Columnar Sum")
          }
        case Count(_) =>
          mode match {
            case Partial | PartialMerge | Final =>
              res_index += 1
            case other =>
              throw new UnsupportedOperationException(
                s"${other} is not supported in Columnar Count")
          }
        case Max(_) =>
          val supportedTypes = List(ByteType, ShortType, IntegerType, LongType,
            FloatType, DoubleType, BooleanType, StringType)
          val max = aggregateFunc.asInstanceOf[Max]
          val aggBufferAttr = max.inputAggBufferAttributes
          val attr = ConverterUtils.getAttrFromExpr(aggBufferAttr.head)
          if (supportedTypes.indexOf(attr.dataType) == -1 &&
              !attr.dataType.isInstanceOf[DecimalType]) {
            throw new UnsupportedOperationException(
              s"${attr.dataType} is not supported in Columnar Max")
          }
          // In native side, DateType is not supported in Max without grouping
          if (groupingExpressions.isEmpty && attr.dataType == DateType) {
            throw new UnsupportedOperationException(
              s"${attr.dataType} is not supported in Columnar Max without grouping")
          }
          mode match {
            case Partial | PartialMerge | Final =>
              res_index += 1
            case other =>
              throw new UnsupportedOperationException(s"not currently supported: $other.")
          }
        case Min(_) =>
          val supportedTypes = List(ByteType, ShortType, IntegerType, LongType,
            FloatType, DoubleType, DateType, BooleanType, StringType)
          val min = aggregateFunc.asInstanceOf[Min]
          val aggBufferAttr = min.inputAggBufferAttributes
          val attr = ConverterUtils.getAttrFromExpr(aggBufferAttr.head)
          if (supportedTypes.indexOf(attr.dataType) == -1 &&
              !attr.dataType.isInstanceOf[DecimalType]) {
            throw new UnsupportedOperationException(
              s"${attr.dataType} is not supported in Columnar Min")
          }
          // DateType is not supported in Min without grouping
          if (groupingExpressions.isEmpty && attr.dataType == DateType) {
            throw new UnsupportedOperationException(
              s"${attr.dataType} is not supported in Columnar Min without grouping")
          }
          mode match {
            case Partial | PartialMerge | Final =>
              res_index += 1
            case other =>
              throw new UnsupportedOperationException(
                s"${other} is not supported in Columnar Min")
          }
        case StddevSamp(_, _) =>
          mode match {
            case Partial =>
              val supportedTypes = List(ByteType, ShortType, IntegerType, LongType,
                FloatType, DoubleType, BooleanType)
              val stddevSamp = aggregateFunc.asInstanceOf[StddevSamp]
              val aggBufferAttr = stddevSamp.inputAggBufferAttributes
              for (index <- aggBufferAttr.indices) {
                val attr = ConverterUtils.getAttrFromExpr(aggBufferAttr(index))
                if (supportedTypes.indexOf(attr.dataType) == -1 &&
                    !attr.dataType.isInstanceOf[DecimalType]) {
                  throw new UnsupportedOperationException(
                    s"${attr.dataType} is not supported in Columnar StddevSampPartial")
                }
              }
              res_index += 3
            case Final =>
              val supportedTypes = List(ByteType, ShortType, IntegerType, LongType,
                FloatType, DoubleType)
              val attr = aggregateAttributeList(res_index)
              if (supportedTypes.indexOf(attr.dataType) == -1) {
                throw new UnsupportedOperationException(
                  s"${attr.dataType} is not supported in Columnar StddevSampFinal")
              }
              res_index += 1
            case other =>
              throw new UnsupportedOperationException(
                s"${other} is not supported in Columnar StddevSamp")
          }
        case other =>
          throw new UnsupportedOperationException(
            s"${other} is not supported in ColumnarAggregation")
      }
    }
  }

  /** ColumnarCodegenSupport **/
  override def inputRDDs(): Seq[RDD[ColumnarBatch]] = child match {
    case c: ColumnarCodegenSupport if c.supportColumnarCodegen == true =>
      c.inputRDDs
    case _ =>
      Seq(child.executeColumnar())
  }

  override def getBuildPlans: Seq[(SparkPlan, SparkPlan)] = child match {
    case c: ColumnarCodegenSupport if c.supportColumnarCodegen == true =>
      c.getBuildPlans
    case _ =>
      Seq()
  }

  override def getStreamedLeafPlan: SparkPlan = child match {
    case c: ColumnarCodegenSupport if c.supportColumnarCodegen == true =>
      c.getStreamedLeafPlan
    case _ =>
      this
  }

  override def updateMetrics(out_num_rows: Long, process_time: Long): Unit = {
    val numOutputRows = longMetric("numOutputRows")
    val procTime = longMetric("processTime")
    procTime.set(process_time / 1000000)
    numOutputRows += out_num_rows
  }

  override def getChild: SparkPlan = child

  override def supportColumnarCodegen: Boolean = true

  // override def canEqual(that: Any): Boolean = false

  def getKernelFunction: TreeNode = {
    ColumnarHashAggregation.prepareKernelFunction(
      groupingExpressions,
      child.output,
      aggregateExpressions,
      aggregateAttributes,
      resultExpressions,
      output,
      sparkConf)
  }

  override def doCodeGen: ColumnarCodegenContext = {

    val childCtx = child match {
      case c: ColumnarCodegenSupport if c.supportColumnarCodegen == true =>
        c.doCodeGen
      case _ =>
        null
    }
    val (codeGenNode, inputSchema) = if (childCtx != null) {
      (
        TreeBuilder.makeFunction(
          s"child",
          Lists.newArrayList(getKernelFunction, childCtx.root),
          new ArrowType.Int(32, true)),
        childCtx.inputSchema)
    } else {
      (
        TreeBuilder.makeFunction(
          s"child",
          Lists.newArrayList(getKernelFunction),
          new ArrowType.Int(32, true)),
        ConverterUtils.toArrowSchema(child.output))
    }
    val outputSchema = ConverterUtils.toArrowSchema(output)
    ColumnarCodegenContext(inputSchema, outputSchema, codeGenNode)
  }

  /****************************/
  override def verboseString(maxFields: Int): String = toString(verbose = true, maxFields)

  override def simpleString(maxFields: Int): String = toString(verbose = false, maxFields)

  private def toString(verbose: Boolean, maxFields: Int): String = {
    val allAggregateExpressions = aggregateExpressions
    val keyString = truncatedString(groupingExpressions, "[", ", ", "]", maxFields)
    val functionString = truncatedString(allAggregateExpressions, "[", ", ", "]", maxFields)
    val outputString = truncatedString(output, "[", ", ", "]", maxFields)
    if (verbose) {
      s"ColumnarHashAggregate(keys=$keyString, functions=$functionString, output=$outputString)"
    } else {
      s"ColumnarHashAggregate(keys=$keyString, functions=$functionString)"
    }
  }
}
