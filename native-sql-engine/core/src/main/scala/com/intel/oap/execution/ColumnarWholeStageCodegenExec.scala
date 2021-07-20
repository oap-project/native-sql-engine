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

import java.util.concurrent.TimeUnit.NANOSECONDS

import com.google.common.collect.Lists
import com.intel.oap.ColumnarPluginConfig
import com.intel.oap.expression._
import com.intel.oap.vectorized.{BatchIterator, ExpressionEvaluator, _}
import org.apache.arrow.gandiva.expression._
import org.apache.arrow.vector.types.pojo.{ArrowType, Field, Schema}

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.datasources.v2.arrow.SparkMemoryUtils
import org.apache.spark.sql.execution.metric.SQLMetrics
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}
import org.apache.spark.util.{ExecutorManager, UserAddedJarUtils}
import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

import org.apache.arrow.dataset.jni.NativeSerializedRecordBatchIterator
import org.apache.arrow.dataset.jni.UnsafeRecordBatchSerializer
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch

case class ColumnarCodegenContext(inputSchema: Schema, outputSchema: Schema, root: TreeNode) {}

trait ColumnarCodegenSupport extends SparkPlan {

  /**
   * Whether this SparkPlan supports whole stage codegen or not.
   */
  def supportColumnarCodegen: Boolean = true

  /**
   * Returns all the RDDs of ColumnarBatch which generates the input rows.
   *
   * @note Right now we support up to two RDDs
   */
  def inputRDDs: Seq[RDD[ColumnarBatch]]

  def getBuildPlans: Seq[(SparkPlan, SparkPlan)]

  def getStreamedLeafPlan: SparkPlan

  def getChild: SparkPlan

  def doCodeGen: ColumnarCodegenContext

  def dependentPlanCtx: ColumnarCodegenContext = null

  def updateMetrics(out_num_rows: Long, process_time: Long): Unit = {}

}

case class ColumnarWholeStageCodegenExec(child: SparkPlan)(val codegenStageId: Int)
    extends UnaryExecNode
    with ColumnarCodegenSupport {

  val sparkConf = sparkContext.getConf
  val numaBindingInfo = ColumnarPluginConfig.getConf.numaBindingInfo

  override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "totalTime" -> SQLMetrics.createTimingMetric(sparkContext, "totaltime_wholestagecodegen"),
    "buildTime" -> SQLMetrics.createTimingMetric(sparkContext, "time to build dependencies"),
    "pipelineTime" -> SQLMetrics.createTimingMetric(sparkContext, "duration"))

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  // This is not strictly needed because the codegen transformation happens after the columnar
  // transformation but just for consistency
  override def supportColumnarCodegen: Boolean = true

  override def supportsColumnar: Boolean = true

  override def otherCopyArgs: Seq[AnyRef] = Seq(codegenStageId.asInstanceOf[Integer])
  override def generateTreeString(
      depth: Int,
      lastChildren: Seq[Boolean],
      append: String => Unit,
      verbose: Boolean,
      prefix: String = "",
      addSuffix: Boolean = false,
      maxFields: Int,
      printNodeId: Boolean,
      indent: Int = 0): Unit = {
    val res = child.generateTreeString(
      depth,
      lastChildren,
      append,
      verbose,
      if (printNodeId) "* " else s"*($codegenStageId) ",
      false,
      maxFields,
      printNodeId,
      indent)
    res
  }

  override def nodeName: String = s"WholeStageCodegenColumnar (${codegenStageId})"
  def uploadAndListJars(signature: String): Seq[String] =
    if (signature != "") {
      if (sparkContext.listJars.filter(path => path.contains(s"${signature}.jar")).isEmpty) {
        val tempDir = ColumnarPluginConfig.getRandomTempDir
        val jarFileName =
          s"${tempDir}/tmp/spark-columnar-plugin-codegen-precompile-${signature}.jar"
        sparkContext.addJar(jarFileName)
      }
      sparkContext.listJars.filter(path => path.contains(s"${signature}.jar"))
    } else {
      Seq()
    }

  override def doCodeGen: ColumnarCodegenContext = {
    val childCtx = child.asInstanceOf[ColumnarCodegenSupport].doCodeGen
    if (childCtx == null) {
      throw new NullPointerException(s"ColumnarWSCG can't doCodeGen on ${child}")
    }
    val wholeStageCodeGenNode = TreeBuilder.makeFunction(
      s"wholestagecodegen",
      Lists.newArrayList(childCtx.root),
      new ArrowType.Int(32, true))
    ColumnarCodegenContext(childCtx.inputSchema, childCtx.outputSchema, wholeStageCodeGenNode)
  }

  override def getBuildPlans: Seq[(SparkPlan, SparkPlan)] = {
    child.asInstanceOf[ColumnarCodegenSupport].getBuildPlans
  }

  override def getStreamedLeafPlan: SparkPlan = {
    child.asInstanceOf[ColumnarCodegenSupport].getStreamedLeafPlan
  }

  override def getChild: SparkPlan = child

  override def updateMetrics(out_num_rows: Long, process_time: Long): Unit = {}

  var metricsUpdated: Boolean = false
  def updateMetrics(nativeIterator: BatchIterator): Unit = {
    if (metricsUpdated == true) return
    try {
      val metrics = nativeIterator.getMetrics
      var curChild = child
      var idx = metrics.output_length_list.size - 1
      var child_process_time: Long = 0
      while (idx >= 0 && curChild.isInstanceOf[ColumnarCodegenSupport]) {
        if (curChild.isInstanceOf[ColumnarConditionProjectExec]) {
          // see if this condition projector did filter, if so, we need to skip metrics
          val condProj = curChild.asInstanceOf[ColumnarConditionProjectExec]
          if (condProj.condition != null && (condProj.projectList != null && condProj.projectList.size != 0)) {
            idx -= 1
          }
        }
        curChild
          .asInstanceOf[ColumnarCodegenSupport]
          .updateMetrics(
            metrics.output_length_list(idx),
            metrics.process_time_list(idx) - child_process_time)
        child_process_time = metrics.process_time_list(idx)
        idx -= 1
        curChild = curChild.asInstanceOf[ColumnarCodegenSupport].getChild
      }
      metricsUpdated = true
    } catch {
      case e: NullPointerException =>
        logWarning(s"updateMetrics failed due to NullPointerException!")
    }
  }

  def prepareRelationFunction(
      keyAttributes: Seq[Attribute],
      outputAttributes: Seq[Attribute]): TreeNode = {
    val outputFieldList: List[Field] = outputAttributes.toList.map(attr => {
      Field
        .nullable(s"${attr.name.toUpperCase()}#${attr.exprId.id}",
        CodeGeneration.getResultType(attr.dataType))
    })

    val keyFieldList: List[Field] = keyAttributes.toList.map(attr => {
      val field = Field
        .nullable(s"${attr.name.toUpperCase()}#${attr.exprId.id}",
          CodeGeneration.getResultType(attr.dataType))
      if (outputFieldList.indexOf(field) == -1) {
        throw new UnsupportedOperationException(s"CachedRelation not found" +
          s"${attr.name.toUpperCase()}#${attr.exprId.id} in ${outputAttributes}")
      }
      field
    })

    val key_args_node = TreeBuilder.makeFunction(
      "key_field",
      keyFieldList
        .map(field => {
          TreeBuilder.makeField(field)
        })
        .asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    val cachedRelationFuncName = "CachedRelation"
    val cached_relation_func = TreeBuilder.makeFunction(
      cachedRelationFuncName,
      Lists.newArrayList(key_args_node),
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    TreeBuilder.makeFunction(
      "standalone",
      Lists.newArrayList(cached_relation_func),
      new ArrowType.Int(32, true))
  }

  /**
   * Return built cpp library's signature
   */
  def doBuild: String = {
    var resCtx: ColumnarCodegenContext = null
    try {
      // call native wholestagecodegen build
      resCtx = doCodeGen
    } catch {
      case e: UnsupportedOperationException
          if e.getMessage == "Unsupport to generate native expression from replaceable expression." =>
        logWarning(e.getMessage())
        ""
      case e: Throwable =>
        throw e
    }
    //resCtx = doCodeGen
    if (resCtx != null) {
      val expression =
        TreeBuilder.makeExpression(
          resCtx.root,
          Field.nullable("result", new ArrowType.Int(32, true)))
      val nativeKernel = new ExpressionEvaluator()
      nativeKernel.build(
        resCtx.inputSchema,
        Lists.newArrayList(expression),
        resCtx.outputSchema,
        true)
    } else {
      ""
    }
  }

  override def inputRDDs(): Seq[RDD[ColumnarBatch]] = child match {
    case c: ColumnarCodegenSupport if c.supportColumnarCodegen == true =>
      c.inputRDDs
    case _ =>
      throw new UnsupportedOperationException
  }
  override def doExecute(): RDD[InternalRow] = {
    throw new UnsupportedOperationException
  }
  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val signature = doBuild
    val listJars = uploadAndListJars(signature)

    val numOutputBatches = child.longMetric("numOutputBatches")
    val pipelineTime = longMetric("pipelineTime")

    var build_elapse: Long = 0
    var eval_elapse: Long = 0
    // we should zip all dependent RDDs to current main RDD
    val buildPlans = getBuildPlans
    val streamedSortPlan = getStreamedLeafPlan
    val contains_aggregate = child.isInstanceOf[ColumnarHashAggregateExec]
    val dependentKernels: ListBuffer[ExpressionEvaluator] = ListBuffer()
    val dependentKernelIterators: ListBuffer[BatchIterator] = ListBuffer()
    val buildRelationBatchHolder: ListBuffer[ColumnarBatch] = ListBuffer()
    val serializableObjectHolder: ListBuffer[SerializableObject] = ListBuffer()
    val relationHolder: ListBuffer[ColumnarHashedRelation] = ListBuffer()
    var idx = 0
    var curRDD = inputRDDs()(0)
    while (idx < buildPlans.length) {

      val curPlan = buildPlans(idx)._1
      val parentPlan = buildPlans(idx)._2

      curRDD = curPlan match {
        case p: ColumnarBroadcastHashJoinExec =>
          val fetchTime = p.longMetric("fetchTime")
          val buildTime = p.longMetric("buildTime")
          val buildPlan = p.getBuildPlan
          val buildInputByteBuf = buildPlan.executeBroadcast[ColumnarHashedRelation]()
          curRDD.mapPartitions { iter =>
            ColumnarPluginConfig.getConf
            ExecutorManager.tryTaskSet(numaBindingInfo)
            // received broadcast value contain a hashmap and raw recordBatch
            val beforeFetch = System.nanoTime()
            val relation = buildInputByteBuf.value.asReadOnlyCopy
            relationHolder += relation
            fetchTime += ((System.nanoTime() - beforeFetch) / 1000000)
            val beforeEval = System.nanoTime()
            val hashRelationObject = relation.hashRelationObj
            serializableObjectHolder += hashRelationObject
            val depIter =
              new CloseableColumnBatchIterator(relation.getColumnarBatchAsIter)
            val ctx = curPlan.asInstanceOf[ColumnarCodegenSupport].dependentPlanCtx
            val expression =
              TreeBuilder.makeExpression(
                ctx.root,
                Field.nullable("result", new ArrowType.Int(32, true)))
            val hashRelationKernel = new ExpressionEvaluator()
            hashRelationKernel.build(ctx.inputSchema, Lists.newArrayList(expression), true)
            val hashRelationResultIterator = hashRelationKernel.finishByIterator()
            dependentKernelIterators += hashRelationResultIterator
            // we need to set original recordBatch to hashRelationKernel
            while (depIter.hasNext) {
              val dep_cb = depIter.next()
              if (dep_cb.numRows > 0) {
                (0 until dep_cb.numCols).toList.foreach(i =>
                  dep_cb.column(i).asInstanceOf[ArrowWritableColumnVector].retain())
                buildRelationBatchHolder += dep_cb
                val dep_rb = ConverterUtils.createArrowRecordBatch(dep_cb)
                hashRelationResultIterator.processAndCacheOne(ctx.inputSchema, dep_rb)
                ConverterUtils.releaseArrowRecordBatch(dep_rb)
              }
            }
            // we need to set hashRelationObject to hashRelationResultIterator
            hashRelationResultIterator.setHashRelationObject(hashRelationObject)
            build_elapse += (System.nanoTime() - beforeEval)
            buildTime += ((System.nanoTime() - beforeEval) / 1000000)
            dependentKernels += hashRelationKernel
            iter
          }
        case p: ColumnarShuffledHashJoinExec =>
          val buildTime = p.longMetric("buildTime")
          val buildPlan = p.getBuildPlan
          curRDD.zipPartitions(buildPlan.executeColumnar()) { (iter, depIter) =>
            ExecutorManager.tryTaskSet(numaBindingInfo)
            val ctx = curPlan.asInstanceOf[ColumnarCodegenSupport].dependentPlanCtx
            val expression =
              TreeBuilder.makeExpression(
                ctx.root,
                Field.nullable("result", new ArrowType.Int(32, true)))
            val hashRelationKernel = new ExpressionEvaluator()
            hashRelationKernel.build(ctx.inputSchema, Lists.newArrayList(expression), true)
            var build_elapse_internal: Long = 0
            while (depIter.hasNext) {
              val dep_cb = depIter.next()
              if (dep_cb.numRows > 0) {
                (0 until dep_cb.numCols).toList.foreach(i =>
                  dep_cb.column(i).asInstanceOf[ArrowWritableColumnVector].retain())
                buildRelationBatchHolder += dep_cb
                val beforeEval = System.nanoTime()
                val dep_rb = ConverterUtils.createArrowRecordBatch(dep_cb)
                hashRelationKernel.evaluate(dep_rb)
                ConverterUtils.releaseArrowRecordBatch(dep_rb)
                build_elapse += System.nanoTime() - beforeEval
                build_elapse_internal += System.nanoTime() - beforeEval
              }
            }
            buildTime += (build_elapse_internal / 1000000)
            dependentKernels += hashRelationKernel
            dependentKernelIterators += hashRelationKernel.finishByIterator()
            iter
          }
        case other =>
          /* we should cache result from this operator */
          curRDD.zipPartitions(other.executeColumnar()) { (iter, depIter) =>
            ExecutorManager.tryTaskSet(numaBindingInfo)
            val curOutput = other match {
              case p: ColumnarSortMergeJoinExec => p.output_skip_alias
              case p: ColumnarBroadcastHashJoinExec => p.output_skip_alias
              case p: ColumnarShuffledHashJoinExec => p.output_skip_alias
              case p => p.output
            }
            val inputSchema = ConverterUtils.toArrowSchema(curOutput)
            val outputSchema = ConverterUtils.toArrowSchema(curOutput)
            if (!parentPlan.isInstanceOf[ColumnarSortMergeJoinExec]) {
              if (parentPlan == null)
                throw new UnsupportedOperationException(
                  s"Only support use ${other.getClass} as buildPlan in ColumnarSortMergeJoin, while this parent Plan is null")
              else
                throw new UnsupportedOperationException(
                  s"Only support use ${other.getClass} as buildPlan in ColumnarSortMergeJoin, while this parent Plan is ${parentPlan.getClass}")
            }
            val parent = parentPlan.asInstanceOf[ColumnarSortMergeJoinExec]
            val keyAttributes =
              if (other.equals(parent.buildPlan))
                parent.buildKeys.map(ConverterUtils.getAttrFromExpr(_))
              else parent.streamedKeys.map(ConverterUtils.getAttrFromExpr(_))
            val cachedFunction = prepareRelationFunction(keyAttributes, curOutput)
            val expression =
              TreeBuilder.makeExpression(
                cachedFunction,
                Field.nullable("result", new ArrowType.Int(32, true)))
            val cachedRelationKernel = new ExpressionEvaluator()
            cachedRelationKernel.build(
              inputSchema,
              Lists.newArrayList(expression),
              outputSchema,
              true)

            // Used as ABI to prevent from serializing buffer data
            val serializedItr: NativeSerializedRecordBatchIterator = {
              new NativeSerializedRecordBatchIterator {

                override def hasNext: Boolean = {
                  depIter.hasNext
                }

                override def next(): Array[Byte] = {
                  val dep_cb = depIter.next()
                  if (dep_cb.numRows > 0) {
                    val dep_rb = ConverterUtils.createArrowRecordBatch(dep_cb)
                    serialize(dep_rb)
                  } else {
                    throw new IllegalStateException()
                  }
                }

                private def serialize(batch: ArrowRecordBatch) = {
                  UnsafeRecordBatchSerializer.serializeUnsafe(batch)
                }

                override def close(): Unit = {
                }
              }
            }
            cachedRelationKernel.evaluate(serializedItr)
            dependentKernels += cachedRelationKernel
            val beforeEval = System.nanoTime()
            dependentKernelIterators += cachedRelationKernel.finishByIterator()
            build_elapse += System.nanoTime() - beforeEval
            iter
          }
      }
      idx += 1
    }

    curRDD.mapPartitions { iter =>
      ExecutorManager.tryTaskSet(numaBindingInfo)
      ColumnarPluginConfig.getConf
      val execTempDir = ColumnarPluginConfig.getTempFile
      val jarList = listJars.map(jarUrl => {
        logWarning(s"Get Codegened library Jar ${jarUrl}")
        UserAddedJarUtils.fetchJarFromSpark(
          jarUrl,
          execTempDir,
          s"spark-columnar-plugin-codegen-precompile-${signature}.jar",
          sparkConf)
        s"${execTempDir}/spark-columnar-plugin-codegen-precompile-${signature}.jar"
      })

      val resCtx = doCodeGen
      val expression =
        TreeBuilder.makeExpression(
          resCtx.root,
          Field.nullable("result", new ArrowType.Int(32, true)))
      val nativeKernel = new ExpressionEvaluator(jarList.toList.asJava)
      nativeKernel.build(
        resCtx.inputSchema,
        Lists.newArrayList(expression),
        resCtx.outputSchema,
        true)
      val nativeIterator = nativeKernel.finishByIterator()
      // we need to complete dependency RDD's firstly
      val beforeBuild = System.nanoTime()
      nativeIterator.setDependencies(dependentKernelIterators.toArray)
      build_elapse += System.nanoTime() - beforeBuild
      val resultStructType = ArrowUtils.fromArrowSchema(resCtx.outputSchema)
      val resIter = streamedSortPlan match {
        case p: ColumnarSortExec =>
          new Iterator[ColumnarBatch] {
            override def hasNext: Boolean = {
              val res = nativeIterator.hasNext
              if (res == false) updateMetrics(nativeIterator)
              res
            }

            override def next(): ColumnarBatch = {
              val beforeEval = System.nanoTime()
              val output_rb = nativeIterator.next
              if (output_rb == null) {
                eval_elapse += System.nanoTime() - beforeEval
                val resultColumnVectors =
                  ArrowWritableColumnVector.allocateColumns(0, resultStructType).toArray
                return new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
              }
              val outputNumRows = output_rb.getLength
              val output = ConverterUtils.fromArrowRecordBatch(resCtx.outputSchema, output_rb)
              ConverterUtils.releaseArrowRecordBatch(output_rb)
              eval_elapse += System.nanoTime() - beforeEval
              new ColumnarBatch(output.map(v => v.asInstanceOf[ColumnVector]), outputNumRows)
            }
          }

        case _ if contains_aggregate =>
          new Iterator[ColumnarBatch] {
            var processed = false
            def process: Unit = {
              while (iter.hasNext) {
                val cb = iter.next()
                if (cb.numRows != 0) {
                  val beforeEval = System.nanoTime()
                  val input_rb =
                    ConverterUtils.createArrowRecordBatch(cb)
                  nativeIterator.processAndCacheOne(resCtx.inputSchema, input_rb)
                  ConverterUtils.releaseArrowRecordBatch(input_rb)
                  eval_elapse += System.nanoTime() - beforeEval
                }
              }
              processed = true
            }
            override def hasNext: Boolean = {
              if (!processed) process
              val res = nativeIterator.hasNext
              if (res == false) updateMetrics(nativeIterator)
              res
            }

            override def next(): ColumnarBatch = {
              if (!processed) process
              val beforeEval = System.nanoTime()
              val output_rb = nativeIterator.next
              if (output_rb == null) {
                eval_elapse += System.nanoTime() - beforeEval
                val resultColumnVectors =
                  ArrowWritableColumnVector.allocateColumns(0, resultStructType).toArray
                return new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
              }
              val outputNumRows = output_rb.getLength
              val output = ConverterUtils.fromArrowRecordBatch(resCtx.outputSchema, output_rb)
              ConverterUtils.releaseArrowRecordBatch(output_rb)
              eval_elapse += System.nanoTime() - beforeEval
              new ColumnarBatch(output.map(v => v.asInstanceOf[ColumnVector]), outputNumRows)
            }
          }

        case _ =>
          // now we can return this wholestagecodegen iter
          new Iterator[ColumnarBatch] {
            override def hasNext: Boolean = {
              val res = iter.hasNext
              if (res == false) updateMetrics(nativeIterator)
              res
            }

            override def next(): ColumnarBatch = {
              val cb = iter.next()
              if (cb.numRows == 0) {
                val resultColumnVectors =
                  ArrowWritableColumnVector.allocateColumns(0, resultStructType).toArray
                return new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
              }
              val beforeEval = System.nanoTime()
              val input_rb =
                ConverterUtils.createArrowRecordBatch(cb)

              val output_rb = nativeIterator.process(resCtx.inputSchema, input_rb)
              if (output_rb == null) {
                ConverterUtils.releaseArrowRecordBatch(input_rb)
                eval_elapse += System.nanoTime() - beforeEval
                val resultColumnVectors =
                  ArrowWritableColumnVector.allocateColumns(0, resultStructType).toArray
                return new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
              }
              val outputNumRows = output_rb.getLength
              ConverterUtils.releaseArrowRecordBatch(input_rb)
              val output = ConverterUtils.fromArrowRecordBatch(resCtx.outputSchema, output_rb)
              ConverterUtils.releaseArrowRecordBatch(output_rb)
              eval_elapse += System.nanoTime() - beforeEval
              new ColumnarBatch(output.map(v => v.asInstanceOf[ColumnVector]), outputNumRows)
            }
          }
      }

      var closed = false
      def close = {
        closed = true
        pipelineTime += (eval_elapse + build_elapse) / 1000000
        buildRelationBatchHolder.foreach(_.close) // fixing: ref cnt goes nagative
        dependentKernels.foreach(_.close)
        dependentKernelIterators.foreach(_.close)
        nativeKernel.close
        nativeIterator.close
        relationHolder.clear()
      }
      SparkMemoryUtils.addLeakSafeTaskCompletionListener[Unit](_ => {
        close
      })
      new CloseableColumnBatchIterator(resIter)
    }
  }
}
