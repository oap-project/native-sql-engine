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

package org.apache.spark.sql.nativesql

import java.io.File

import com.intel.oap.spark.sql.DataFrameReaderImplicits._
import com.intel.oap.spark.sql.execution.datasources.v2.arrow.ArrowOptions
import org.apache.spark.SparkConf
import org.apache.spark.sql.{IntegratedUDFTestUtils, QueryTest}
import org.apache.spark.sql.execution.ArrowColumnarCachedBatchSerializer
import org.apache.spark.sql.internal.{SQLConf, StaticSQLConf}
import org.apache.spark.sql.internal.StaticSQLConf.SPARK_SESSION_EXTENSIONS
import org.apache.spark.sql.test.SharedSparkSession

class NativeArrowDataSourceTest extends QueryTest with SharedSparkSession {
  import IntegratedUDFTestUtils._

  val pyarrowTestUDF: TestScalarPandasUDF = TestScalarPandasUDF(name = "pyarrowUDF")
  val parquetFile = "part-01999-ee647adf-a7c9-4193-abff-e3b1ce037256-c000.snappy.parquet"

  override protected def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.memory.offHeap.size", String.valueOf(300 * 1024 * 1024))
    conf.set(SPARK_SESSION_EXTENSIONS.key,
      "com.intel.oap.ColumnarPlugin, com.intel.oap.spark.sql.ArrowWriteExtension")
      .set(
        StaticSQLConf.SPARK_CACHE_SERIALIZER.key,
        classOf[ArrowColumnarCachedBatchSerializer].getName)
    conf
  }

  test("arrow_udf test") {
    val path = NativeArrowDataSourceTest.locateResourcePath(parquetFile)
    val df = spark.read.option(ArrowOptions.KEY_ORIGINAL_FORMAT, "parquet").arrow(path)
    val df1 = df.withColumn("p_a", pyarrowTestUDF(df("cs_sold_date_sk")))
    logWarning(s"physical plan is:\n ${df1.queryExecution.executedPlan}")
    df1.show()
  }

  test("arrow_udf test with filter") {
    val path = NativeArrowDataSourceTest.locateResourcePath(parquetFile)
    spark.read.option(ArrowOptions.KEY_ORIGINAL_FORMAT, "parquet").arrow(path)
      .createOrReplaceTempView("tmp")
    val df1 = sql("select cs_sold_date_sk, cs_sold_time_sk from tmp " +
      "where cs_bill_customer_sk is not null")
    val df2 = df1.withColumn("p_a", pyarrowTestUDF(df1("cs_sold_date_sk")))
    logWarning(s"physical plan is:\n ${df2.queryExecution.executedPlan}")
    df2.show()
  }
}

object NativeArrowDataSourceTest {
  private def locateResourcePath(resource: String): String = {
    "src/test/resources/test-data/".concat(File.separator).concat(resource)
  }
}
