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

import com.intel.oap.expression.ConverterUtils
import com.intel.oap.vectorized.{ArrowColumnarToRowJniWrapper, ArrowWritableColumnVector}
import org.apache.arrow.vector.types.pojo.{Field, Schema}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder, UnsafeRow}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.datasources.v2.arrow.SparkMemoryUtils
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer
import scala.concurrent.duration._

case class ArrowColumnarToRowExec(child: SparkPlan) extends UnaryExecNode {
  override def nodeName: String = "ArrowColumnarToRow"

  assert(child.supportsColumnar)

  override def output: Seq[Attribute] = child.output

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  buildCheck()

  def buildCheck(): Unit = {
    val schema = child.schema
    for (field <- schema.fields) {
      try {
        ConverterUtils.checkIfTypeSupported(field.dataType)
      } catch {
        case e: UnsupportedOperationException =>
          throw new UnsupportedOperationException(
            s"${field.dataType} is not supported in ArrowColumnarToRowExec.")
      }
    }
  }

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "number of input batches"),
    "convertTime" -> SQLMetrics.createTimingMetric(sparkContext, "time to convert"),
    "hasNextTime" -> SQLMetrics.createTimingMetric(sparkContext, "time to has next"),
    "nextTime" -> SQLMetrics.createTimingMetric(sparkContext, "time to next")
  )

  override def doExecute(): RDD[InternalRow] = {
    val numOutputRows = longMetric("numOutputRows")
    val numInputBatches = longMetric("numInputBatches")
    val convertTime = longMetric("convertTime")
    val hasNextTime = longMetric("hasNextTime")
    val nextTime = longMetric("nextTime")

    child.executeColumnar().mapPartitions { batches =>
      // TODO:: pass the jni jniWrapper and arrowSchema  and serializeSchema method by broadcast
      val jniWrapper = new ArrowColumnarToRowJniWrapper()
      var arrowSchema: Array[Byte] = null

      def serializeSchema(fields: Seq[Field]): Array[Byte] = {
        val schema = new Schema(fields.asJava)
        ConverterUtils.getSchemaBytesBuf(schema)
      }

      batches.flatMap { batch =>
        numInputBatches += 1
        numOutputRows += batch.numRows()

        if (batch.numRows == 0 || batch.numCols == 0) {
          logInfo(s"Skip ColumnarBatch of ${batch.numRows} rows, ${batch.numCols} cols")
          Iterator.empty
        } else {
          val bufAddrs = new ListBuffer[Long]()
          val bufSizes = new ListBuffer[Long]()
          val fields = new ListBuffer[Field]()
          (0 until batch.numCols).foreach { idx =>
            val column = batch.column(idx).asInstanceOf[ArrowWritableColumnVector]
            fields += column.getValueVector.getField
            column.getValueVector
              .getBuffers(false)
              .foreach { buffer =>
                bufAddrs += buffer.memoryAddress()
                bufSizes += buffer.readableBytes()
              }
          }

          if (arrowSchema == null) {
            arrowSchema = serializeSchema(fields)
          }

          val beforeConvert = System.nanoTime()

          val instanceID = jniWrapper.nativeConvertColumnarToRow(
            arrowSchema, batch.numRows, bufAddrs.toArray, bufSizes.toArray,
            SparkMemoryUtils.contextMemoryPool().getNativeInstanceId)

          convertTime += NANOSECONDS.toMillis(System.nanoTime() - beforeConvert)

          new Iterator[InternalRow] {
            override def hasNext: Boolean = {
              val beforeHasNext = System.nanoTime()
              val result = jniWrapper.nativeHasNext(instanceID)

              hasNextTime += NANOSECONDS.toMillis(System.nanoTime() - beforeHasNext)
              result

            }
            override def next: UnsafeRow = {
              val beforeNext = System.nanoTime()
              val row = jniWrapper.nativeNext(instanceID)
              nextTime += NANOSECONDS.toMillis(System.nanoTime() - beforeNext)
              row
            }
          }
        }
      }
   }
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[ArrowColumnarToRowExec]

  override def equals(other: Any): Boolean = other match {
    case that: ArrowColumnarToRowExec =>
      (that canEqual this) && super.equals(that)
    case _ => false
  }
}