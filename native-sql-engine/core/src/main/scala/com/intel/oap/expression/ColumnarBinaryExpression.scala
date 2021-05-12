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

import com.google.common.collect.Lists
import org.apache.arrow.gandiva.evaluator._
import org.apache.arrow.gandiva.exceptions.GandivaException
import org.apache.arrow.gandiva.expression._
import org.apache.arrow.vector.types.pojo.ArrowType
import org.apache.arrow.vector.types.pojo.Field
import org.apache.arrow.vector.types.IntervalUnit
import org.apache.arrow.vector.types.DateUnit
import org.apache.arrow.vector.types.pojo.ArrowType.ArrowTypeID
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer

/**
 * A version of add that supports columnar processing for longs.
 */
class ColumnarDateAddInterval(start: Expression, interval: Expression, original: DateAddInterval)
    extends DateAddInterval(start, interval, original.timeZoneId, original.ansiEnabled)
    with ColumnarExpression
    with Logging {
  override def doColumnarCodeGen(args: Object): (TreeNode, ArrowType) = {
    var (left_node, left_type): (TreeNode, ArrowType) =
      left.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)
    var (right_node, right_type): (TreeNode, ArrowType) =
      right.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = left_type
    if (right_type.getTypeID != ArrowTypeID.Interval) {
      throw new IllegalArgumentException(
        "ColumnarDateAddInterval requires for Interval type on second parameter")
    }
    right_type.asInstanceOf[ArrowType.Interval].getUnit match {
      case IntervalUnit.DAY_TIME =>
        (
          TreeBuilder.makeFunction("add", Lists.newArrayList(left_node, right_node), resultType),
          resultType)
      case IntervalUnit.YEAR_MONTH =>
        (
          TreeBuilder.makeFunction(
            "dateAddYearMonthInterval",
            Lists.newArrayList(left_node, right_node),
            resultType),
          resultType)
      case _ =>
        throw new IllegalArgumentException(
          "ColumnarDateAddInterval only support interval type as DATE_TIME or YEAR_MONTH")
    }

  }
}

class ColumnarUnixTimestamp(timeExp: Expression, format: Expression, original: UnixTimestamp)
  extends UnixTimestamp(timeExp, format, original.timeZoneId)
    with ColumnarExpression
    with Logging {
  override def doColumnarCodeGen(args: Object): (TreeNode, ArrowType) = {
    var (timeExp_node, timeExp_type): (TreeNode, ArrowType) =
      timeExp.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)
    var (format_node, format_type): (TreeNode, ArrowType) =
      format.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    if (timeExp_type.isInstanceOf[ArrowType.Date]) {
      val limitLen: java.lang.Long = 10L
      val limitLenNode = TreeBuilder.makeLiteral(limitLen)
      timeExp_node =
        TreeBuilder.makeFunction("castVARCHAR", Lists.newArrayList(timeExp_node, limitLenNode), ArrowType.Utf8.INSTANCE)
    }
    val resultType = new ArrowType.Date(DateUnit.MILLISECOND)
    val castNode =
      TreeBuilder.makeFunction("to_date", Lists.newArrayList(timeExp_node, format_node), resultType)
    (castNode, resultType)
  }
}

object ColumnarBinaryExpression {

  def create(left: Expression, right: Expression, original: Expression): Expression =
    original match {
      case s: DateAddInterval =>
        new ColumnarDateAddInterval(left, right, s)
      case u: UnixTimestamp =>
        new ColumnarUnixTimestamp(left, right, u)
      case other =>
        throw new UnsupportedOperationException(s"not currently supported: $other.")
    }
}
