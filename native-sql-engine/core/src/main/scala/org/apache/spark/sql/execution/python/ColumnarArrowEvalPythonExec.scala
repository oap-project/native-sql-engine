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

package org.apache.spark.sql.execution.python

import java.io._
import java.net._
import java.util.concurrent.atomic.AtomicBoolean

import com.intel.oap.expression._
import com.intel.oap.vectorized._

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

import org.apache.arrow.vector.{ValueVector, VectorLoader, VectorSchemaRoot}
import org.apache.arrow.vector.ipc.{ArrowStreamReader, ArrowStreamWriter}
import org.apache.spark.{ContextAwareIterator, SparkEnv}
import org.apache.spark.TaskContext
import org.apache.spark.api.python.{BasePythonRunner, ChainedPythonFunctions, PythonRDD, SpecialLengths}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution.python.{EvalPythonExec, PythonArrowOutput, PythonUDFRunner}
import org.apache.spark.sql.execution.datasources.v2.arrow.SparkMemoryUtils
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector, ArrowColumnVector}
import org.apache.spark.sql.types.{DataType, StructField, StructType}
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

/**
 * Similar to `PythonUDFRunner`, but exchange data with Python worker via Arrow stream.
 */
class ColumnarArrowPythonRunner(
    funcs: Seq[ChainedPythonFunctions],
    evalType: Int,
    argOffsets: Array[Array[Int]],
    schema: StructType,
    timeZoneId: String,
    conf: Map[String, String])
  extends BasePythonRunner[ColumnarBatch, ColumnarBatch](funcs, evalType, argOffsets) {

  override val simplifiedTraceback: Boolean = SQLConf.get.pysparkSimplifiedTraceback

  override val bufferSize: Int = SQLConf.get.pandasUDFBufferSize
  require(
    bufferSize >= 4,
    "Pandas execution requires more than 4 bytes. Please set higher buffer. " +
      s"Please change '${SQLConf.PANDAS_UDF_BUFFER_SIZE.key}'.")

  protected def newReaderIterator(
      stream: DataInputStream,
      writerThread: WriterThread,
      startTime: Long,
      env: SparkEnv,
      worker: Socket,
      releasedOrClosed: AtomicBoolean,
      context: TaskContext): Iterator[ColumnarBatch] = {

    new ReaderIterator(stream, writerThread, startTime, env, worker, releasedOrClosed, context) {
      private val allocator = ArrowUtils.rootAllocator.newChildAllocator(
        s"stdin reader for $pythonExec", 0, Long.MaxValue)

      private var reader: ArrowStreamReader = _
      private var root: VectorSchemaRoot = _
      private var schema: StructType = _
      private var vectors: Array[ColumnVector] = _

      context.addTaskCompletionListener[Unit] { _ =>
        if (reader != null) {
          reader.close(false)
        }
        allocator.close()
      }

      private var batchLoaded = true

      protected override def read(): ColumnarBatch = {
        if (writerThread.exception.isDefined) {
          throw writerThread.exception.get
        }
        try {
          if (reader != null && batchLoaded) {
            batchLoaded = reader.loadNextBatch()
            if (batchLoaded) {
              val batch = new ColumnarBatch(vectors)
              batch.setNumRows(root.getRowCount)
              batch
            } else {
              reader.close(false)
              allocator.close()
              // Reach end of stream. Call `read()` again to read control data.
              read()
            }
          } else {
            stream.readInt() match {
              case SpecialLengths.START_ARROW_STREAM =>
                reader = new ArrowStreamReader(stream, allocator)
                root = reader.getVectorSchemaRoot()
                schema = ArrowUtils.fromArrowSchema(root.getSchema())
                vectors = ArrowWritableColumnVector.loadColumns(root.getRowCount, root.getFieldVectors).toArray[ColumnVector]
                read()
              case SpecialLengths.TIMING_DATA =>
                handleTimingData()
                read()
              case SpecialLengths.PYTHON_EXCEPTION_THROWN =>
                throw handlePythonException()
              case SpecialLengths.END_OF_DATA_SECTION =>
                handleEndOfDataSection()
                null
            }
          }
        } catch handleException
      }
    }
  }

  protected override def newWriterThread(
      env: SparkEnv,
      worker: Socket,
      inputIterator: Iterator[ColumnarBatch],
      partitionIndex: Int,
      context: TaskContext): WriterThread = {
    new WriterThread(env, worker, inputIterator, partitionIndex, context) {

      protected override def writeCommand(dataOut: DataOutputStream): Unit = {

        // Write config for the worker as a number of key -> value pairs of strings
        dataOut.writeInt(conf.size)
        for ((k, v) <- conf) {
          PythonRDD.writeUTF(k, dataOut)
          PythonRDD.writeUTF(v, dataOut)
        }

        PythonUDFRunner.writeUDFs(dataOut, funcs, argOffsets)
      }

      protected override def writeIteratorToStream(dataOut: DataOutputStream): Unit = {
        var numRows: Long = 0
        val arrowSchema = ArrowUtils.toArrowSchema(schema, timeZoneId)
        val allocator = ArrowUtils.rootAllocator.newChildAllocator(
          s"stdout writer for $pythonExec", 0, Long.MaxValue)
        val root = VectorSchemaRoot.create(arrowSchema, allocator)

        Utils.tryWithSafeFinally {
          val loader = new VectorLoader(root)
          val writer = new ArrowStreamWriter(root, null, dataOut)
          writer.start()
//          var lastBatch: ColumnarBatch = null
          while (inputIterator.hasNext) {
            val nextBatch = inputIterator.next()
//            if (lastBatch != null) {
//              (0 until lastBatch.numCols).toList.foreach(i =>
//                logWarning(s"last refCnt: " +
//                  s"${lastBatch.column(i).asInstanceOf[ArrowWritableColumnVector].refCnt()}"))
//            }
//            (0 until nextBatch.numCols).toList.foreach(i =>
//              logWarning(s"next refCnt: " +
//                s"${nextBatch.column(i).asInstanceOf[ArrowWritableColumnVector].refCnt()}"))
//            lastBatch = nextBatch
            numRows += nextBatch.numRows
            val next_rb = ConverterUtils.createArrowRecordBatch(nextBatch)
            loader.load(next_rb)
            writer.writeBatch()
            ConverterUtils.releaseArrowRecordBatch(next_rb)
          }
          // end writes footer to the output stream and doesn't clean any resources.
          // It could throw exception if the output stream is closed, so it should be
          // in the try block.
          writer.end()
        }{
          root.close()
          allocator.close()
        }
      }
    }
  }

}

/**
 * Grouped a iterator into batches.
 * This is similar to iter.grouped but returns Iterator[T] instead of Seq[T].
 * This is necessary because sometimes we cannot hold reference of input rows
 * because the some input rows are mutable and can be reused.
 */
private[spark] class CoalecseBatchIterator[T](iter: Iterator[T], batchSize: Int)
  extends Iterator[T] {

  override def hasNext: Boolean = iter.hasNext

  override def next(): T = {
    iter.next
  }
}

case class ColumnarArrowEvalPythonExec(udfs: Seq[PythonUDF], resultAttrs: Seq[Attribute], child: SparkPlan,
    evalType: Int)
    extends EvalPythonExec {
  override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "output_batches"),
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "input_batches"),
    "processTime" -> SQLMetrics.createTimingMetric(sparkContext, "totaltime_arrow_udf"))
  
  private val batchSize = conf.arrowMaxRecordsPerBatch
  private val sessionLocalTimeZone = conf.sessionLocalTimeZone
  private val pythonRunnerConf = ArrowUtils.getPythonRunnerConfMap(conf)

  override def supportsColumnar = true
  
  protected def evaluate(
      funcs: Seq[ChainedPythonFunctions],
      argOffsets: Array[Array[Int]],
      iter: Iterator[InternalRow],
      schema: StructType,
      context: TaskContext): Iterator[InternalRow] = {
    throw new NotImplementedError("evaluate Internal row is not supported")
  }

  protected def evaluateColumnar(
    funcs: Seq[ChainedPythonFunctions],
    argOffsets: Array[Array[Int]],
    iter: Iterator[ColumnarBatch],
    schema: StructType,
    context: TaskContext): Iterator[ColumnarBatch] = {

    val outputTypes = output.drop(child.output.length).map(_.dataType)

    // Use Coalecse to improve performance in future
    // val batchIter = new CoalecseBatchIterator(iter, batchSize)
    val batchIter = new CloseableColumnBatchIterator(iter)

    val columnarBatchIter = new ColumnarArrowPythonRunner(
      funcs,
      evalType,
      argOffsets,
      schema,
      sessionLocalTimeZone,
      pythonRunnerConf).compute(batchIter, context.partitionId(), context)

    columnarBatchIter.map { batch =>
      val actualDataTypes = (0 until batch.numCols()).map(i => batch.column(i).dataType())
      assert(outputTypes == actualDataTypes, "Invalid schema from arrow_udf: " +
        s"expected ${outputTypes.mkString(", ")}, got ${actualDataTypes.mkString(", ")}")
      batch
    }
  }

  private def collectFunctions(udf: PythonUDF): (ChainedPythonFunctions, Seq[Expression]) = {
    udf.children match {
      case Seq(u: PythonUDF) =>
        val (chained, children) = collectFunctions(u)
        (ChainedPythonFunctions(chained.funcs ++ Seq(udf.func)), children)
      case children =>
        // There should not be any other UDFs, or the children can't be evaluated directly.
        assert(children.forall(_.find(_.isInstanceOf[PythonUDF]).isEmpty))
        (ChainedPythonFunctions(Seq(udf.func)), udf.children)
    }
  }

  protected override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = longMetric("numOutputRows")
    val numOutputBatches = longMetric("numOutputBatches")
    val numInputBatches = longMetric("numInputBatches")
    val procTime = longMetric("processTime")

    val inputRDD = child.executeColumnar()
    inputRDD.mapPartitions { iter =>
      val context = TaskContext.get()
      val contextAwareIterator = new ContextAwareIterator(context, iter)

      val (pyFuncs, inputs) = udfs.map(collectFunctions).unzip

      // flatten all the arguments
      val allInputs = new ArrayBuffer[Expression]
      val dataTypes = new ArrayBuffer[DataType]
      val argOffsets = inputs.map { input =>
        input.map { e =>
          if (allInputs.exists(_.semanticEquals(e))) {
            allInputs.indexWhere(_.semanticEquals(e))
          } else {
            allInputs += e
            dataTypes += e.dataType
            allInputs.length - 1
          }
        }.toArray
      }.toArray

      // start to work on input data and data type
      val projector = ColumnarProjection.create(child.output, allInputs.toSeq)
      val projected_ordinal_list = projector.getOrdinalList
      val schema = StructType(dataTypes.zipWithIndex.map { case (dt, i) =>
        StructField(s"_$i", dt)
      }.toSeq)
      val input_cb_cache = new ArrayBuffer[ColumnarBatch]()

      // original spark will cache input into files using HybridRowQueue
      // we will retain columnar batch in memory firstly
      var start_time: Long = 0
      val projectedColumnarBatchIter = contextAwareIterator.map { input_cb =>
        numInputBatches += input_cb.numRows
        start_time = System.nanoTime()
        // 0. cache input for later merge
        (0 until input_cb.numCols).foreach(i => {
          input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].retain()
        })
        input_cb_cache += input_cb
        // 1. doing projection to input
        val valueVectors = (0 until input_cb.numCols).toList.map(i =>
              input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].getValueVector())
        val ret = if (projector.needEvaluate) {
          val projectedInput = projector.evaluate(input_cb.numRows, valueVectors)
          new ColumnarBatch(projectedInput.toArray, input_cb.numRows)
        } else {
          // for no-need project evaluate, do another retain
          (0 until input_cb.numCols).foreach(i => {
            input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].retain()
          })
          new ColumnarBatch(projected_ordinal_list.toArray.map(i => input_cb.column(i)), input_cb.numRows)
        }
        ret
      }.map(batch => {
        val actualDataTypes = (0 until batch.numCols()).map(i => batch.column(i).dataType())
        assert(dataTypes == actualDataTypes, "Invalid schema for arrow_udf: " +
          s"expected ${dataTypes.mkString(", ")}, got ${actualDataTypes.mkString(", ")}")
        batch
      })
      SparkMemoryUtils.addLeakSafeTaskCompletionListener[Unit]((tc: TaskContext) => {
        projector.close
        input_cb_cache.foreach(_.close)
      })

      val outputColumnarBatchIterator = evaluateColumnar(
        pyFuncs, argOffsets, projectedColumnarBatchIter, schema, context)

      // val resultProj = ColumnarProjection.create(output, output)

      new CloseableColumnBatchIterator(
        outputColumnarBatchIterator.zipWithIndex.map { case (output_cb, batchId) =>
          val input_cb = input_cb_cache(batchId)
          // retain for input_cb since we are passing it to next operator
//          (0 until input_cb.numCols).toList.foreach(i =>
//            logWarning(s"before retain refCnt: ${input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].refCnt()}"))
          (0 until input_cb.numCols).toList.foreach(i => {
            input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].retain()
          })
//          (0 until input_cb.numCols).toList.foreach(i =>
//            logWarning(s"after retain refCnt: ${input_cb.column(i).asInstanceOf[ArrowWritableColumnVector].refCnt()}"))
          val joinedVectors = (0 until input_cb.numCols).toArray.map(i => input_cb.column(i)) ++ (0 until output_cb.numCols).toArray.map(i => output_cb.column(i))
          val numRows = input_cb.numRows
          numOutputBatches += 1
          numOutputRows += numRows
          procTime += (System.nanoTime() - start_time) / 1000000
          new ColumnarBatch(joinedVectors, numRows)
          /* below is for in case there will be some scala projection in demand
          val valueVectors = joinedVectors.toList.map(_.asInstanceOf[ArrowWritableColumnVector].getValueVector())
          val projectedOutput = resultProj.evaluate(numRows, valueVectors)
          new ColumnarBatch(projectedOutput.toArray.map(_.asInstanceOf[ColumnVector]), numRows)*/
        }
      )
    }
  }

  protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
    copy(child = newChild)
}
