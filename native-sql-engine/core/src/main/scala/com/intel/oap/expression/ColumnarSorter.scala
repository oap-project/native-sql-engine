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

import java.nio.ByteBuffer
import java.util.concurrent.TimeUnit._
import com.google.common.collect.Lists

import com.intel.oap.ColumnarPluginConfig
import com.intel.oap.vectorized.ArrowWritableColumnVector
import com.intel.oap.vectorized.ExpressionEvaluator
import com.intel.oap.vectorized.BatchIterator

import org.apache.spark.internal.Logging
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}
import org.apache.spark.sql.types._
import org.apache.spark.TaskContext

import org.apache.arrow.gandiva.evaluator._
import org.apache.arrow.gandiva.exceptions.GandivaException
import org.apache.arrow.gandiva.expression._
import org.apache.arrow.vector._
import org.apache.arrow.vector.ipc.message.ArrowFieldNode
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.arrow.vector.types.pojo.Field
import org.apache.arrow.vector.types.pojo.ArrowType

import scala.collection.Iterator
import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import scala.collection.immutable.List
import scala.collection.mutable.ArrayBuffer

class ColumnarSorter(
    sorter: ExpressionEvaluator,
    outputAttributes: Seq[Attribute],
    sortTime: SQLMetric,
    outputBatches: SQLMetric,
    outputRows: SQLMetric,
    shuffleTime: SQLMetric,
    elapse: SQLMetric,
    sparkConf: SparkConf)
    extends Logging {
  var processedNumRows: Long = 0
  var sort_elapse: Long = 0
  var shuffle_elapse: Long = 0
  var total_elapse: Long = 0
  val inputBatchHolder = new ListBuffer[ColumnarBatch]()
  var nextVector: FieldVector = _
  var closed: Boolean = false
  val resultSchema: StructType = StructType(
    outputAttributes
      .map(expr => {
        val attr = ConverterUtils.getAttrFromExpr(expr)
        StructField(s"${attr.name.toLowerCase()}", attr.dataType, nullable = true)
      })
      .toArray)
  val outputFieldList: List[Field] = outputAttributes.toList.map(expr => {
    val attr = ConverterUtils.getAttrFromExpr(expr)
    Field.nullable(s"${attr.name.toLowerCase()}#${attr.exprId.id}",
        CodeGeneration.getResultType(attr.dataType))
  })
  val arrowSchema = new Schema(outputFieldList.asJava)
  var sort_iterator: BatchIterator = _

  def close(): Unit = {
    if (closed) return
    logInfo(s"Sort Closed, ${processedNumRows} rows, output ${outputRows} rows")
    if (nextVector != null) {
      nextVector.close()
    }
    elapse.set(NANOSECONDS.toMillis(total_elapse))
    sortTime.set(NANOSECONDS.toMillis(sort_elapse))
    shuffleTime.set(NANOSECONDS.toMillis(shuffle_elapse))
    //inputBatchHolder.foreach(cb => cb.close())
    //inputBatchHolder.clear
    if (sorter != null) {
      sorter.close()
    }
    if (sort_iterator != null) {
      sort_iterator.close()
      sort_iterator = null
    }
    closed = true
  }

  def updateSorterResult(input: ColumnarBatch): Unit = {
    inputBatchHolder += input
    val input_batch = ConverterUtils.createArrowRecordBatch(input)
    (0 until input.numCols).toList.foreach(i =>
      input.column(i).asInstanceOf[ArrowWritableColumnVector].retain())
    val beforeSort = System.nanoTime()
    sorter.evaluate2(input_batch)
    sort_elapse += System.nanoTime() - beforeSort
    total_elapse += System.nanoTime() - beforeSort
    ConverterUtils.releaseArrowRecordBatch(input_batch)
  }

  def getSorterResult(resultBatch: ArrowRecordBatch): ColumnarBatch = {
    if (resultBatch == null) {
      val resultColumnVectors =
        ArrowWritableColumnVector.allocateColumns(0, resultSchema).toArray
      new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
    } else {
      val resultColumnVectorList =
        ConverterUtils.fromArrowRecordBatch(arrowSchema, resultBatch)
      val length = resultBatch.getLength()
      ConverterUtils.releaseArrowRecordBatch(resultBatch)
      new ColumnarBatch(resultColumnVectorList.map(v => v.asInstanceOf[ColumnVector]), length)
    }
  }

  def createColumnarIterator(cbIterator: Iterator[ColumnarBatch]): Iterator[ColumnarBatch] = {
    new Iterator[ColumnarBatch] {
      var cb: ColumnarBatch = null
      var resultCb: ColumnarBatch = null
      var nextBatch: ArrowRecordBatch = null
      var batchIterator: BatchIterator = null
      var has_next: Boolean = true

      override def hasNext: Boolean = {
        if (has_next && sort_iterator == null) {
          while (cbIterator.hasNext) {
            cb = cbIterator.next()

            if (cb.numRows > 0) {
              updateSorterResult(cb)
              processedNumRows += cb.numRows
            }
          }

          val beforeSort = System.nanoTime()
          if (processedNumRows > 0) {
            sort_iterator = sorter.finishByIterator();
          }
          sort_elapse += System.nanoTime() - beforeSort
          total_elapse += System.nanoTime() - beforeSort
        }
        if (sort_iterator != null && sort_iterator.hasNext()) {
          return true
        } else {
          has_next = false
          //inputBatchHolder.foreach(cb => cb.close())
          //inputBatchHolder.clear
          return false
        }
      }

      override def next(): ColumnarBatch = {
        val beforeShuffle = System.nanoTime()
        nextBatch = sort_iterator.next()
        resultCb = getSorterResult(nextBatch)
        shuffle_elapse += System.nanoTime() - beforeShuffle
        total_elapse += System.nanoTime() - beforeShuffle
        outputBatches += 1
        outputRows += nextBatch.getLength()
        resultCb
      }
    }
  }
}

object ColumnarSorter extends Logging {

  def checkIfKeyFound(sortOrder: Seq[SortOrder], outputAttributes: Seq[Attribute]): Unit = {
    val outputFieldList: List[Field] = outputAttributes.toList.map(expr => {
      val attr = ConverterUtils.getAttrFromExpr(expr)
      Field.nullable(s"${attr.name.toLowerCase()}#${attr.exprId.id}",
        CodeGeneration.getResultType(attr.dataType))
    })
    sortOrder.toList.foreach(sort => {
      val attr = ConverterUtils.getAttrFromExpr(sort.child)
      val field = Field.nullable(s"${attr.name.toLowerCase()}#${attr.exprId.id}",
        CodeGeneration.getResultType(attr.dataType))
      if (outputFieldList.indexOf(field) == -1) {
        throw new UnsupportedOperationException(
          s"ColumnarSorter not found ${attr.name.toLowerCase()}#${attr.exprId.id} " +
            s"in ${outputAttributes}")
      }
    })
  }

  def prepareRelationFunction(
      sortOrder: Seq[SortOrder],
      outputAttributes: Seq[Attribute]): TreeNode = {
    checkIfKeyFound(sortOrder, outputAttributes)
    val keyFieldList: List[Field] = sortOrder.toList.map(sort => {
      val attr = ConverterUtils.getAttrFromExpr(sort.child)
      Field.nullable(s"${attr.name.toLowerCase()}#${attr.exprId.id}",
        CodeGeneration.getResultType(attr.dataType))
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

  def prepareKernelFunction(
      sortOrder: Seq[SortOrder],
      outputAttributes: Seq[Attribute],
      sparkConf: SparkConf,
      result_type: Int = 0): TreeNode = {
    logInfo(s"ColumnarSorter sortOrder is ${sortOrder}, outputAttributes is ${outputAttributes}")
    checkIfKeyFound(sortOrder, outputAttributes)
    val NaNCheck = ColumnarPluginConfig.getConf.enableColumnarNaNCheck
    val codegen = ColumnarPluginConfig.getConf.enableColumnarCodegenSort
    /////////////// Prepare ColumnarSorter //////////////
    val keyFieldList: List[Field] = sortOrder.toList.map(sort => {
      val attr = ConverterUtils.getAttrFromExpr(sort.child)
      Field.nullable(s"${attr.name.toLowerCase()}#${attr.exprId.id}",
        CodeGeneration.getResultType(attr.dataType))
    })

    /*
    Get the sort directions and nulls order from SortOrder.
    Directions: asc: true, desc: false
    NullsOrder: NullsFirst: true, NullsLast: false
    */
    var directions = new ListBuffer[Boolean]()
    var nullsOrder = new ListBuffer[Boolean]()
    for (key <- sortOrder) {
      val asc = key.isAscending
      val nullOrdering = key.nullOrdering
      directions += asc
      nullsOrder += {
        nullOrdering match {
          case NullsFirst => true
          case NullsLast => false
        }
      }
    }
    val dirList = directions.toList
    val nullList = nullsOrder.toList

    val sortKeyFuncList: List[TreeNode] = sortOrder.toList.map(expr => {
      val (nativeNode, returnType) = ConverterUtils.getColumnarFuncNode(expr.child)
      if (s"${nativeNode.toProtobuf}".contains("none#")) {
        throw new UnsupportedOperationException(
          s"Unsupport to generate native expression from replaceable expression.")
      }
      nativeNode
    })

    val sort_keys_node = TreeBuilder.makeFunction(
      "key_function",
      sortKeyFuncList.asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    val key_args_node = TreeBuilder.makeFunction(
      "key_field",
      keyFieldList
        .map(field => {
          TreeBuilder.makeField(field)
        })
        .asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    val dir_node = TreeBuilder.makeFunction(
      "sort_directions",
      dirList
        .map(dir => {
          TreeBuilder.makeLiteral(dir.asInstanceOf[java.lang.Boolean])
        })
        .asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    val nulls_order_node = TreeBuilder.makeFunction(
      "sort_nulls_order",
      nullList
        .map(nullsOrder => {
          TreeBuilder.makeLiteral(nullsOrder.asInstanceOf[java.lang.Boolean])
        })
        .asJava,
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    val NaN_check_node = TreeBuilder.makeFunction(
      "NaN_check",
      Lists.newArrayList(
        TreeBuilder.makeLiteral(NaNCheck.asInstanceOf[java.lang.Boolean])),
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    val codegen_node = TreeBuilder.makeFunction(
      "codegen",
      Lists.newArrayList(TreeBuilder.makeLiteral(codegen.asInstanceOf[java.lang.Boolean])),
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    val result_type_node = TreeBuilder.makeFunction(
      "result_type",
      Lists.newArrayList(TreeBuilder.makeLiteral(result_type.asInstanceOf[Integer])),
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    val sortFuncName = "sortArraysToIndices"
    val sort_func_node = TreeBuilder.makeFunction(
      sortFuncName,
      Lists
        .newArrayList(
          sort_keys_node,
          key_args_node,
          dir_node,
          nulls_order_node,
          NaN_check_node,
          codegen_node,
          result_type_node),
      new ArrowType.Int(32, true) /*dummy ret type, won't be used*/ )

    TreeBuilder.makeFunction(
      "standalone",
      Lists.newArrayList(sort_func_node),
      new ArrowType.Int(32, true))
  }

  def init(
      sortOrder: Seq[SortOrder],
      outputAttributes: Seq[Attribute],
      _sparkConf: SparkConf): (ExpressionTree, Schema) = {
    val outputFieldList: List[Field] = outputAttributes.toList.map(expr => {
      val attr = ConverterUtils.getAttrFromExpr(expr)
      Field.nullable(s"${attr.name.toLowerCase()}#${attr.exprId.id}",
          CodeGeneration.getResultType(attr.dataType))
    })
    val retType = Field.nullable("res", new ArrowType.Int(32, true))
    val sort_node =
      prepareKernelFunction(sortOrder, outputAttributes, _sparkConf)

    (TreeBuilder.makeExpression(sort_node, retType), new Schema(outputFieldList.asJava))
  }

  def prebuild(
      sortOrder: Seq[SortOrder],
      outputAttributes: Seq[Attribute],
      sortTime: SQLMetric,
      outputBatches: SQLMetric,
      outputRows: SQLMetric,
      shuffleTime: SQLMetric,
      elapse: SQLMetric,
      sparkConf: SparkConf): String = synchronized {
    val (sort_expr, arrowSchema) = init(sortOrder, outputAttributes, sparkConf)
    val sorter = new ExpressionEvaluator()
    val signature = sorter
      .build(arrowSchema, Lists.newArrayList(sort_expr), arrowSchema, true /*return at finish*/)
    sorter.close
    signature
  }

  def create(
      sortOrder: Seq[SortOrder],
      outputAttributes: Seq[Attribute],
      listJars: Seq[String],
      sortTime: SQLMetric,
      outputBatches: SQLMetric,
      outputRows: SQLMetric,
      shuffleTime: SQLMetric,
      elapse: SQLMetric,
      sparkConf: SparkConf): ColumnarSorter = synchronized {
    val (sort_expr, arrowSchema) = init(sortOrder, outputAttributes, sparkConf)
    val sorter = new ExpressionEvaluator(listJars.toList.asJava)
    sorter
      .build(arrowSchema, Lists.newArrayList(sort_expr), arrowSchema, true /*return at finish*/, true/*enable spill*/)
    new ColumnarSorter(
      sorter,
      outputAttributes,
      sortTime,
      outputBatches,
      outputRows,
      shuffleTime,
      elapse,
      sparkConf)
  }

}
