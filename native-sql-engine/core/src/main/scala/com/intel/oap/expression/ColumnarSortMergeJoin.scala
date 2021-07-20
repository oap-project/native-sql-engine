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

package com.intel.oap.expression

import java.util.concurrent.TimeUnit._

import com.intel.oap.ColumnarPluginConfig
import com.intel.oap.vectorized.ArrowWritableColumnVector
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.codegen._
import org.apache.spark.sql.catalyst.expressions.BindReferences.bindReferences
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution.{BinaryExecNode, CodegenSupport, SparkPlan}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.catalyst.optimizer.{BuildLeft, BuildRight, BuildSide}

import scala.collection.JavaConverters._
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}

import scala.collection.mutable.ListBuffer
import org.apache.arrow.vector.ipc.message.ArrowFieldNode
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch
import org.apache.arrow.vector.types.pojo.ArrowType
import org.apache.arrow.vector.types.pojo.Field
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.arrow.gandiva.expression._
import org.apache.arrow.gandiva.evaluator._
import org.apache.arrow.memory.ArrowBuf
import com.google.common.collect.Lists
import org.apache.spark.sql.types.{DataType, DecimalType, StructType}
import com.intel.oap.vectorized.ExpressionEvaluator
import com.intel.oap.vectorized.BatchIterator
import org.apache.spark.sql.util.ArrowUtils

/**
 * Performs a sort merge join of two child relations.
 */
class ColumnarSortMergeJoin(
    prober: ExpressionEvaluator,
    build_input_arrow_schema: Schema,
    stream_input_arrow_schema: Schema,
    output_arrow_schema: Schema,
    leftKeys: Seq[Expression],
    rightKeys: Seq[Expression],
    resultSchema: StructType,
    joinType: JoinType,
    conditionOption: Option[Expression],
    left: SparkPlan,
    right: SparkPlan,
    isSkewJoin: Boolean,
    joinTime: SQLMetric,
    prepareTime: SQLMetric,
    totaltime_sortmergejoin: SQLMetric,
    totalOutputNumRows: SQLMetric,
    sparkConf: SparkConf,
    buildProjector: ColumnarProjection,
    buildKeyProjectOrdinalList: List[Int],
    streamProjector: ColumnarProjection,
    streamKeyProjectOrdinalList: List[Int])
    extends Logging {
  ColumnarPluginConfig.getConf
  var probe_iterator: BatchIterator = _
  var build_cb: ColumnarBatch = null
  var last_cb: ColumnarBatch = null

  val inputBatchHolder = new ListBuffer[ColumnarBatch]()

  def columnarJoin(
      streamIter: Iterator[ColumnarBatch],
      buildIter: Iterator[ColumnarBatch]): Iterator[ColumnarBatch] = {

    val (realbuildIter, realstreamIter) = joinType match {
      case LeftSemi =>
        (streamIter, buildIter)
      case LeftOuter =>
        (streamIter, buildIter)
      case LeftAnti =>
        (streamIter, buildIter)
      case ExistenceJoin(_) =>
        (streamIter, buildIter)
      case _ =>
        (buildIter, streamIter)
    }

    while (realbuildIter.hasNext) {
      if (build_cb != null) {
        build_cb = null
      }
      build_cb = realbuildIter.next()
      val beforeBuild = System.nanoTime()
      val projectedBuildKeyCols: List[ArrowWritableColumnVector] = if (buildProjector != null) {
        val builderOrdinalList = buildProjector.getOrdinalList()
        val builderAttributes = buildProjector.output()
        val builderProjectCols = builderOrdinalList.map(i => {
          build_cb.column(i).asInstanceOf[ArrowWritableColumnVector]
        })
        buildProjector.evaluate(build_cb.numRows, builderProjectCols.map(_.getValueVector()))
      } else {
        List[ArrowWritableColumnVector]()
      }
      val buildCols = (0 until build_cb.numCols).toList.map(i =>
        build_cb.column(i).asInstanceOf[ArrowWritableColumnVector]) ::: projectedBuildKeyCols
      val build_rb =
        ConverterUtils.createArrowRecordBatch(build_cb.numRows, buildCols.map(_.getValueVector))

      buildCols.indices.toList.foreach(i =>
        buildCols(i).retain())
      inputBatchHolder += build_cb
      prober.evaluate(build_rb)
      prepareTime += NANOSECONDS.toMillis(System.nanoTime() - beforeBuild)
      ConverterUtils.releaseArrowRecordBatch(build_rb)
      projectedBuildKeyCols.foreach(v => v.close())
    }
    if (build_cb != null) {
      build_cb = null
    } else {
      if ((joinType == LeftOuter || joinType == RightOuter) && realstreamIter.hasNext) {
        // If stream side still has next batch, an empty batch is assigned to build side.
        val resultColumnVectors = ArrowWritableColumnVector
          .allocateColumns(0, ArrowUtils.fromArrowSchema(build_input_arrow_schema))
          .toArray
        build_cb =
          new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
        val beforeBuild = System.nanoTime()
        val projectedBuildKeyCols: List[ArrowWritableColumnVector] = if (buildProjector != null) {
          val builderOrdinalList = buildProjector.getOrdinalList()
          val builderAttributes = buildProjector.output()
          val builderProjectCols = builderOrdinalList.map(i => {
            build_cb.column(i).asInstanceOf[ArrowWritableColumnVector]
          })
          buildProjector.evaluate(build_cb.numRows, builderProjectCols.map(_.getValueVector()))
        } else {
          List[ArrowWritableColumnVector]()
        }
        val buildCols = (0 until build_cb.numCols).toList.map(i =>
          build_cb.column(i).asInstanceOf[ArrowWritableColumnVector]) ::: projectedBuildKeyCols
        val build_rb =
          ConverterUtils.createArrowRecordBatch(build_cb.numRows, buildCols.map(_.getValueVector))
        buildCols.indices.toList.foreach(i => buildCols(i).retain())
        inputBatchHolder += build_cb
        prober.evaluate(build_rb)
        prepareTime += NANOSECONDS.toMillis(System.nanoTime() - beforeBuild)
        ConverterUtils.releaseArrowRecordBatch(build_rb)
        projectedBuildKeyCols.foreach(v => v.close)
      } else {
        val res = new Iterator[ColumnarBatch] {
          override def hasNext: Boolean = {
            false
          }
          override def next(): ColumnarBatch = {
            val resultColumnVectors = ArrowWritableColumnVector
              .allocateColumns(0, resultSchema)
              .toArray
            new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
          }
        }
        return res
      }
    }
    val beforeBuild = System.nanoTime()
    probe_iterator = prober.finishByIterator()
    prepareTime += NANOSECONDS.toMillis(System.nanoTime() - beforeBuild)

    new Iterator[ColumnarBatch] {
      override def hasNext: Boolean = {
        if (realstreamIter.hasNext) {
          true
        } else {
          inputBatchHolder.foreach(cb => cb.close())
          false
        }
      }

      override def next(): ColumnarBatch = {
        val cb = realstreamIter.next()
        last_cb = cb
        val beforeJoin = System.nanoTime()
        
        val output_rb = if (cb.numRows > 0) {
          val projectedStreamKeyCols: List[ArrowWritableColumnVector] =
            if (streamProjector != null) {
              val streamOrdinalList = streamProjector.getOrdinalList
              val streamAttributes = streamProjector.output
              val streamProjectCols = streamOrdinalList.map(i => {
                cb.column(i).asInstanceOf[ArrowWritableColumnVector]
              })
              streamProjector.evaluate(cb.numRows, streamProjectCols.map(_.getValueVector()))
            } else {
              List[ArrowWritableColumnVector]()
            }
          val streamCols = (0 until cb.numCols).toList.map(i =>
            cb.column(i).asInstanceOf[ArrowWritableColumnVector]) ::: projectedStreamKeyCols
          val stream_rb: ArrowRecordBatch =
            ConverterUtils.createArrowRecordBatch(cb.numRows, streamCols.map(_.getValueVector))

          val res = probe_iterator.process(stream_input_arrow_schema, stream_rb)

          ConverterUtils.releaseArrowRecordBatch(stream_rb)
          projectedStreamKeyCols.foreach(v => v.close)
          res
        } else {
          null
        }
        joinTime += NANOSECONDS.toMillis(System.nanoTime() - beforeJoin)
        if (output_rb == null) {
          val resultColumnVectors =
            ArrowWritableColumnVector.allocateColumns(0, resultSchema).toArray
          new ColumnarBatch(resultColumnVectors.map(v => v.asInstanceOf[ColumnVector]).toArray, 0)
        } else {
          val outputNumRows = output_rb.getLength
          val output = ConverterUtils.fromArrowRecordBatch(output_arrow_schema, output_rb)
          ConverterUtils.releaseArrowRecordBatch(output_rb)
          totalOutputNumRows += outputNumRows
          new ColumnarBatch(output.map(v => v.asInstanceOf[ColumnVector]).toArray, outputNumRows)
        }
      }
    }
  }

  def close(): Unit = {
    if (prober != null) {
      prober.close()
    }
    if (probe_iterator != null) {
      probe_iterator.close()
      probe_iterator = null
    }
    totaltime_sortmergejoin.merge(prepareTime)
    totaltime_sortmergejoin.merge(joinTime)
  }
}

object ColumnarSortMergeJoin extends Logging {
  var columnarSortMergeJoin: ColumnarSortMergeJoin = _
  var build_input_arrow_schema: Schema = _
  var stream_input_arrow_schema: Schema = _
  var output_arrow_schema: Schema = _
  var condition_probe_expr: ExpressionTree = _
  var prober: ExpressionEvaluator = _
  var buildKeyProjectOrdinalList: List[Int] = _
  var streamKeyProjectOrdinalList: List[Int] = _

  def init(
      leftKeys: Seq[Expression],
      rightKeys: Seq[Expression],
      resultSchema: StructType,
      joinType: JoinType,
      conditionOption: Option[Expression],
      left: SparkPlan,
      right: SparkPlan,
      _joinTime: SQLMetric,
      _prepareTime: SQLMetric,
      _totaltime_sortmergejoin: SQLMetric,
      _numOutputRows: SQLMetric,
      _sparkConf: SparkConf): Unit = {
    val joinTime = _joinTime
    val prepareTime = _prepareTime
    val totaltime_sortmergejoin = _totaltime_sortmergejoin
    val numOutputRows = _numOutputRows
    val sparkConf = _sparkConf
    ColumnarPluginConfig.getConf
    // TODO
    val l_input_schema: List[Attribute] = left.output.toList
    val r_input_schema: List[Attribute] = right.output.toList
    logInfo(
      s"\nleft_schema is ${l_input_schema}, right_schema is ${r_input_schema}, \nleftKeys is ${leftKeys}, rightKeys is ${rightKeys}, \nresultSchema is ${resultSchema}, \njoinType is ${joinType}, condition is ${conditionOption}")

    val l_input_field_list: List[Field] = l_input_schema.toList.map(attr => {
      Field
        .nullable(s"${attr.name}#${attr.exprId.id}", CodeGeneration.getResultType(attr.dataType))
    })
    val r_input_field_list: List[Field] = r_input_schema.toList.map(attr => {
      Field
        .nullable(s"${attr.name}#${attr.exprId.id}", CodeGeneration.getResultType(attr.dataType))
    })

    val resultFieldList = resultSchema
      .map(field => {
        Field.nullable(field.name, CodeGeneration.getResultType(field.dataType))
      })
      .toList

    logInfo(s"leftKeyExpression is ${leftKeys}, rightKeyExpression is ${rightKeys}")
    val lkeyProjectOrdinalList = new ListBuffer[Int]()
    val lkeyFieldList: List[Field] = leftKeys.toList.zipWithIndex.map {
      case (expr, i) => {
        //TODO(): fix this workaround
        //if (expr.isInstanceOf[AttributeReference] && expr.asInstanceOf[AttributeReference].name == "none") {
        //  return
        //}
        val (nativeNode, returnType) = ConverterUtils.getColumnarFuncNode(expr)
        if (s"${nativeNode.toProtobuf}".contains("none#")) {
          throw new UnsupportedOperationException(
            s"Unsupport to generate native expression from replaceable expression.")
        }
        if (s"${nativeNode.toProtobuf}".contains("fnNode")) {
          //throw new UnsupportedOperationException(s"join key with expression is not supported.")
          lkeyProjectOrdinalList += i
          Field.nullable(s"${expr}", returnType)
        } else {
          val attr = ConverterUtils.getAttrFromExpr(expr)
          Field
            .nullable(
              s"${attr.name}#${attr.exprId.id}",
              CodeGeneration.getResultType(attr.dataType))
        }
      }
    }

    val rkeyProjectOrdinalList = new ListBuffer[Int]()
    val rkeyFieldList: List[Field] = rightKeys.toList.zipWithIndex.map {
      case (expr, i) => {
        val (nativeNode, returnType) = ConverterUtils.getColumnarFuncNode(expr)
        if (s"${nativeNode.toProtobuf}".contains("fnNode")) {
          //throw new UnsupportedOperationException(s"join key with expression is not supported.")
          rkeyProjectOrdinalList += i
          Field.nullable(s"${expr}", returnType)
        } else {
          val attr = ConverterUtils.getAttrFromExpr(expr)
          Field
            .nullable(
              s"${attr.name}#${attr.exprId.id}",
              CodeGeneration.getResultType(attr.dataType))
        }
      }
    }

    //TODO: fix join left/right
    val buildSide: BuildSide = joinType match {
      case LeftSemi =>
        BuildRight
      case LeftOuter =>
        BuildRight
      case LeftAnti =>
        BuildRight
      case j: ExistenceJoin =>
        BuildRight
      case LeftExistence(_) =>
        BuildRight
      case _ =>
        BuildLeft
    }
    var (
      build_key_field_list,
      stream_key_field_list,
      build_input_field_list,
      stream_input_field_list) = buildSide match {
      case BuildLeft =>
        buildKeyProjectOrdinalList = lkeyProjectOrdinalList.toList
        streamKeyProjectOrdinalList = rkeyProjectOrdinalList.toList
        (lkeyFieldList, rkeyFieldList, l_input_field_list, r_input_field_list)

      case BuildRight =>
        buildKeyProjectOrdinalList = rkeyProjectOrdinalList.toList
        streamKeyProjectOrdinalList = lkeyProjectOrdinalList.toList
        (rkeyFieldList, lkeyFieldList, r_input_field_list, l_input_field_list)

    }

    var existsField: Field = null
    var existsIndex: Int = -1
    val (probe_func_name, build_output_field_list, stream_output_field_list) = joinType match {
      case _: InnerLike =>
        ("conditionedJoinArraysInner", build_input_field_list, stream_input_field_list)
      case LeftSemi =>
        ("conditionedJoinArraysSemi", List[Field](), stream_input_field_list)
      case LeftOuter =>
        ("conditionedJoinArraysOuter", build_input_field_list, stream_input_field_list)
      case RightOuter =>
        ("conditionedJoinArraysOuter", build_input_field_list, stream_input_field_list)
      //TODO: full outer is not supported yet
      //case FullOuter =>
      //  ("conditionedJoinArraysFullOuter", build_input_field_list, stream_input_field_list)
      case LeftAnti =>
        ("conditionedJoinArraysAnti", List[Field](), stream_input_field_list)
      case j: ExistenceJoin =>
        val existsSchema = j.exists
        existsField = Field.nullable(
          s"${existsSchema.name}",
          CodeGeneration.getResultType(existsSchema.dataType))
        // Use the last index for we cannot distinguish between two "exists" StructType
        existsIndex = resultFieldList.lastIndexOf(existsField)
        existsField = Field.nullable(
          s"${existsSchema.name}#${existsSchema.exprId.id}",
          CodeGeneration.getResultType(existsSchema.dataType))
        ("conditionedJoinArraysExistence", build_input_field_list, stream_input_field_list)
      case _ =>
        throw new UnsupportedOperationException(s"Join Type ${joinType} is not supported yet.")
    }

    build_input_arrow_schema = new Schema(build_input_field_list.asJava)

    stream_input_arrow_schema = new Schema(stream_input_field_list.asJava)

    val condition = conditionOption match {
      case Some(c) =>
        c
      case None =>
        null
    }
    val (conditionInputFieldList, conditionOutputFieldList) = buildSide match {
      case BuildLeft =>
        joinType match {
          case ExistenceJoin(_) =>
            throw new UnsupportedOperationException(
              s"BuildLeft for ${joinType} is not supported yet.")
          case _ =>
            (build_input_field_list, build_output_field_list ::: stream_output_field_list)
        }
      case BuildRight =>
        joinType match {
          case ExistenceJoin(_) =>
            val (front, back) = stream_output_field_list.splitAt(existsIndex)
            val existsOutputFieldList = (front :+ existsField) ::: back
            (build_input_field_list, existsOutputFieldList)
          case _ =>
            (build_input_field_list, stream_output_field_list ::: build_output_field_list)
        }
    }
    // we need to add projectedKeyOutput into input_field_list here
    if (buildKeyProjectOrdinalList.nonEmpty) {
      build_input_field_list =
        build_input_field_list ::: buildKeyProjectOrdinalList.map(i => build_key_field_list(i))
    }

    if (streamKeyProjectOrdinalList.nonEmpty) {
      stream_input_field_list =
        stream_input_field_list ::: streamKeyProjectOrdinalList.map(i => stream_key_field_list(i))
    }
    build_input_arrow_schema = new Schema(build_input_field_list.asJava)
    stream_input_arrow_schema = new Schema(stream_input_field_list.asJava)

    val conditionArrowSchema = new Schema(conditionInputFieldList.asJava)
    output_arrow_schema = new Schema(conditionOutputFieldList.asJava)
    var conditionInputList: java.util.List[Field] = Lists.newArrayList()
    val build_args_node = TreeBuilder.makeFunction(
      "codegen_left_schema",
      build_input_field_list
        .map(field => {
          TreeBuilder.makeField(field)
        })
        .asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val stream_args_node = TreeBuilder.makeFunction(
      "codegen_right_schema",
      stream_input_field_list
        .map(field => {
          TreeBuilder.makeField(field)
        })
        .asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val build_keys_node = TreeBuilder.makeFunction(
      "codegen_left_key_schema",
      build_key_field_list
        .map(field => {
          TreeBuilder.makeField(field)
        })
        .asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val stream_keys_node = TreeBuilder.makeFunction(
      "codegen_right_key_schema",
      stream_key_field_list
        .map(field => {
          TreeBuilder.makeField(field)
        })
        .asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val condition_expression_node_list
        : java.util.List[org.apache.arrow.gandiva.expression.TreeNode] =
      if (condition != null) {
        val columnarExpression: Expression =
          ColumnarExpressionConverter.replaceWithColumnarExpression(condition)
        val (condition_expression_node, resultType) =
          columnarExpression
            .asInstanceOf[ColumnarExpression]
            .doColumnarCodeGen(conditionInputList)
        Lists.newArrayList(build_keys_node, stream_keys_node, condition_expression_node)
      } else {
        Lists.newArrayList(build_keys_node, stream_keys_node)
      }
    val retType = Field.nullable("res", new ArrowType.Int(32, true))

    // Make Expresion for conditionedProbe
    val condition_probe_node = TreeBuilder.makeFunction(
      probe_func_name,
      condition_expression_node_list,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val codegen_probe_node = TreeBuilder.makeFunction(
      "codegen_withTwoInputs",
      Lists.newArrayList(condition_probe_node, build_args_node, stream_args_node),
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    condition_probe_expr = TreeBuilder.makeExpression(codegen_probe_node, retType)
  }

  def precheck(
      leftKeys: Seq[Expression],
      rightKeys: Seq[Expression],
      resultSchema: StructType,
      joinType: JoinType,
      condition: Option[Expression],
      left: SparkPlan,
      right: SparkPlan,
      joinTime: SQLMetric,
      prepareTime: SQLMetric,
      totaltime_sortmergejoin: SQLMetric,
      numOutputRows: SQLMetric,
      sparkConf: SparkConf): Unit = synchronized {
    logInfo("precheck")
    init(
      leftKeys,
      rightKeys,
      resultSchema,
      joinType,
      condition,
      left,
      right,
      joinTime,
      prepareTime,
      totaltime_sortmergejoin,
      numOutputRows,
      sparkConf)

  }

  def prebuild(
      leftKeys: Seq[Expression],
      rightKeys: Seq[Expression],
      resultSchema: StructType,
      joinType: JoinType,
      condition: Option[Expression],
      left: SparkPlan,
      right: SparkPlan,
      joinTime: SQLMetric,
      prepareTime: SQLMetric,
      totaltime_sortmergejoin: SQLMetric,
      numOutputRows: SQLMetric,
      sparkConf: SparkConf): String = synchronized {
    logInfo("prebuild")
    init(
      leftKeys,
      rightKeys,
      resultSchema,
      joinType,
      condition,
      left,
      right,
      joinTime,
      prepareTime,
      totaltime_sortmergejoin,
      numOutputRows,
      sparkConf)

    prober = new ExpressionEvaluator()
    val signature = prober.build(
      build_input_arrow_schema,
      Lists.newArrayList(condition_probe_expr),
      output_arrow_schema,
      true)
    prober.close
    signature
  }

  def create(
      leftKeys: Seq[Expression],
      rightKeys: Seq[Expression],
      resultSchema: StructType,
      joinType: JoinType,
      condition: Option[Expression],
      left: SparkPlan,
      right: SparkPlan,
      isSkewJoin: Boolean,
      listJars: Seq[String],
      joinTime: SQLMetric,
      prepareTime: SQLMetric,
      totaltime_sortmergejoin: SQLMetric,
      numOutputRows: SQLMetric,
      sparkConf: SparkConf): ColumnarSortMergeJoin = synchronized {

    init(
      leftKeys,
      rightKeys,
      resultSchema,
      joinType,
      condition,
      left,
      right,
      joinTime,
      prepareTime,
      totaltime_sortmergejoin,
      numOutputRows,
      sparkConf)

    val buildSide: BuildSide = joinType match {
      case LeftSemi =>
        BuildRight
      case LeftOuter =>
        BuildRight
      case LeftAnti =>
        BuildRight
      case j: ExistenceJoin =>
        BuildRight
      case LeftExistence(_) =>
        BuildRight
      case _ =>
        BuildLeft
    }

    val (buildProjector, streamProjector) =
      // create gandiva project to pre-process
      buildSide match {
        case BuildLeft =>
          (
            (if (buildKeyProjectOrdinalList.nonEmpty)
               ColumnarProjection
                 .create(left.output, buildKeyProjectOrdinalList.map(i => leftKeys(i)))
             else null),
            (if (streamKeyProjectOrdinalList.nonEmpty)
               ColumnarProjection
                 .create(right.output, streamKeyProjectOrdinalList.map(i => rightKeys(i)))
             else null))

        case BuildRight =>
          (
            (if (buildKeyProjectOrdinalList.nonEmpty)
               ColumnarProjection
                 .create(right.output, buildKeyProjectOrdinalList.map(i => rightKeys(i)))
             else null),
            (if (streamKeyProjectOrdinalList.nonEmpty)
               ColumnarProjection
                 .create(left.output, streamKeyProjectOrdinalList.map(i => leftKeys(i)))
             else null))
      }

    prober = new ExpressionEvaluator(listJars.toList.asJava)
    prober.build(
      build_input_arrow_schema,
      Lists.newArrayList(condition_probe_expr),
      output_arrow_schema,
      true)

    new ColumnarSortMergeJoin(
      prober,
      build_input_arrow_schema,
      stream_input_arrow_schema,
      output_arrow_schema,
      leftKeys,
      rightKeys,
      resultSchema,
      joinType,
      condition,
      left,
      right,
      isSkewJoin,
      joinTime,
      prepareTime,
      totaltime_sortmergejoin,
      numOutputRows,
      sparkConf,
      buildProjector,
      buildKeyProjectOrdinalList,
      streamProjector,
      streamKeyProjectOrdinalList)
    
  }

  def close(): Unit = {
    if (columnarSortMergeJoin != null) {
      columnarSortMergeJoin.close()
    }
  }

  def prepareKernelFunction(
      buildKeys: Seq[Expression],
      streamedKeys: Seq[Expression],
      buildInputAttributes: Seq[Attribute],
      streamedInputAttributes: Seq[Attribute],
      output: Seq[Attribute],
      joinType: JoinType,
      conditionOption: Option[Expression]): TreeNode = {
    /////// Build side ///////
    val buildInputFieldList: List[Field] = buildInputAttributes.toList.map(attr => {
      Field
        .nullable(s"${attr.name}#${attr.exprId.id}", CodeGeneration.getResultType(attr.dataType))
    })
    val buildKeysFunctionList: List[TreeNode] = buildKeys.toList.map(expr => {
      val (nativeNode, returnType) = ConverterUtils.getColumnarFuncNode(expr)
      if (s"${nativeNode.toProtobuf}".contains("none#")) {
        throw new UnsupportedOperationException(
          s"Unsupport to generate native expression from replaceable expression.")
      }
      nativeNode
    })
    /////// Streamed side ///////
    val streamedInputFieldList: List[Field] = streamedInputAttributes.toList.map(attr => {
      Field
        .nullable(s"${attr.name}#${attr.exprId.id}", CodeGeneration.getResultType(attr.dataType))
    })
    val streamedKeysFunctionList: List[TreeNode] = streamedKeys.toList.map(expr => {
      val (nativeNode, returnType) = ConverterUtils.getColumnarFuncNode(expr)
      if (s"${nativeNode.toProtobuf}".contains("none#")) {
        throw new UnsupportedOperationException(
          s"Unsupport to generate native expression from replaceable expression.")
      }
      nativeNode
    })
    ///////////////////////////////////

    val resultFunctionList: List[TreeNode] = output.toList.map(field => {
      val field_node = Field.nullable(
        s"${field.name}#${field.exprId.id}",
        CodeGeneration.getResultType(field.dataType))
      TreeBuilder.makeField(field_node)
    })
    ///////////////////////////////////

    var existsField: Field = null
    var existsIndex: Int = -1
    val probeFuncName = joinType match {
      case _: InnerLike =>
        "conditionedMergeJoinInner"
      case LeftSemi =>
        "conditionedMergeJoinSemi"
      case LeftOuter =>
        "conditionedMergeJoinOuter"
      case RightOuter =>
        "conditionedMergeJoinOuter"
      case LeftAnti =>
        "conditionedMergeJoinAnti"
      case j: ExistenceJoin =>
        "conditionedMergeJoinExistence"
      case _ =>
        throw new UnsupportedOperationException(s"Join Type ${joinType} is not supported yet.")
    }

    val condition = conditionOption.getOrElse(null)
    var conditionInputList: java.util.List[Field] = Lists.newArrayList()
    val build_args_node = TreeBuilder.makeFunction(
      "codegen_left_schema",
      buildInputFieldList.map(field => { TreeBuilder.makeField(field) }).asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val stream_args_node = TreeBuilder.makeFunction(
      "codegen_right_schema",
      streamedInputFieldList.map(field => { TreeBuilder.makeField(field) }).asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val build_keys_node = TreeBuilder.makeFunction(
      "codegen_left_key_schema",
      buildKeysFunctionList.asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val stream_keys_node = TreeBuilder.makeFunction(
      "codegen_right_key_schema",
      streamedKeysFunctionList.asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val result_node = TreeBuilder.makeFunction(
      "codegen_result_schema",
      resultFunctionList.asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
    val condition_expression_node_list: java.util.List[TreeNode] =
      if (condition != null) {
        val columnarExpression: Expression =
          ColumnarExpressionConverter.replaceWithColumnarExpression(condition)
        val (condition_expression_node, resultType) =
          columnarExpression
            .asInstanceOf[ColumnarExpression]
            .doColumnarCodeGen(conditionInputList)
        Lists.newArrayList(
          build_args_node,
          stream_args_node,
          build_keys_node,
          stream_keys_node,
          result_node,
          condition_expression_node)
      } else {
        Lists.newArrayList(
          build_args_node,
          stream_args_node,
          build_keys_node,
          stream_keys_node,
          result_node)
      }
    val retType = Field.nullable("res", new ArrowType.Int(32, true))

    // Make Expresion for conditionedProbe
    TreeBuilder.makeFunction(
      probeFuncName,
      condition_expression_node_list,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )
  }
}
