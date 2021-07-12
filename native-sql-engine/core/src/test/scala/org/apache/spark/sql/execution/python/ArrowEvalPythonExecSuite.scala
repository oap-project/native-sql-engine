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

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.api.python.{PythonEvalType, PythonFunction}
import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.expressions.{And, AttributeReference, GreaterThan, In}
import org.apache.spark.sql.execution.{FilterExec, InputAdapter, SparkPlanTest, WholeStageCodegenExec}
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.{IntegratedUDFTestUtils, QueryTest}
import org.apache.spark.sql.Row
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.types.{StructType, StructField, IntegerType}

class ArrowEvalPythonExecSuite extends QueryTest with SharedSparkSession {

  import testImplicits.newProductEncoder
  import testImplicits.localSeqToDatasetHolder
  import IntegratedUDFTestUtils._

  val pyarrowTestUDF = TestScalarPandasUDF(name = "pyarrowUDF")

  lazy val base = Seq(
    (1, 1), (1, 2), (2, 1), (2, 2),
    (3, 1), (3, 2), (0, 1), (3, 0)).toDF("a", "b")

  test("arrow_udf test") {
    lazy val expected = Seq(
      (0, 1, 0), (1, 1, 1), (1, 2, 1), (2, 1, 2), (2, 2, 2), (3, 0, 3), (3, 1, 3), (3, 2, 3)
      ).toDF("a", "b", "p_a")

    val df2 = base.withColumn("p_a", pyarrowTestUDF(base("a")))
    checkAnswer(df2, expected)
  }
}