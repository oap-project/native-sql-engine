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
import org.apache.arrow.vector.types.{DateUnit, FloatingPointPrecision, TimeUnit}
import org.apache.arrow.vector.types.pojo.Field
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.optimizer._
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer

/**
 * A version of add that supports columnar processing for longs.
 */
class ColumnarIsNotNull(child: Expression, original: Expression)
    extends IsNotNull(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(
      ByteType,
      ShortType,
      IntegerType,
      LongType,
      FloatType,
      DoubleType,
      DateType,
      TimestampType,
      BooleanType,
      StringType,
      BinaryType)
    if (supportedTypes.indexOf(child.dataType) == -1 &&
        !child.dataType.isInstanceOf[DecimalType]) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarIsNotNull.")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = new ArrowType.Bool()
    val funcNode =
      TreeBuilder.makeFunction("isnotnull", Lists.newArrayList(child_node), resultType)
    (funcNode, resultType)
  }
}

class ColumnarIsNull(child: Expression, original: Expression)
    extends IsNotNull(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(
      ByteType,
      ShortType,
      IntegerType,
      LongType,
      FloatType,
      DoubleType,
      DateType,
      TimestampType,
      BooleanType,
      StringType,
      BinaryType)
    if (supportedTypes.indexOf(child.dataType) == -1 &&
        !child.dataType.isInstanceOf[DecimalType]) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarIsNull.")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = new ArrowType.Bool()
    val funcNode =
      TreeBuilder.makeFunction("isnull", Lists.newArrayList(child_node), resultType)
    (funcNode, resultType)
  }
}

class ColumnarMonth(child: Expression, original: Expression)
    extends Month(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(LongType, StringType, DateType)
    if (supportedTypes.indexOf(child.dataType) == -1) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarMonth.")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = new ArrowType.Int(32, true)
    //FIXME(): requires utf8()/int64() as input
    val cast_func =
      TreeBuilder.makeFunction(
        "castDATE",
        Lists.newArrayList(child_node),
        new ArrowType.Date(DateUnit.MILLISECOND))
    val funcNode =
      TreeBuilder.makeFunction(
        "extractMonth",
        Lists.newArrayList(cast_func),
        new ArrowType.Int(64, true))
    val castNode =
      TreeBuilder.makeFunction("castINT", Lists.newArrayList(funcNode), resultType)
    (castNode, resultType)
  }
}

class ColumnarDayOfMonth(child: Expression, original: Expression)
  extends DayOfMonth(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(LongType, StringType, DateType)
    if (supportedTypes.indexOf(child.dataType) == -1) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarDayOfMonth.")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = new ArrowType.Int(32, true)
    //FIXME(): requires utf8()/int64() as input
    val cast_func =
    TreeBuilder.makeFunction(
      "castDATE",
      Lists.newArrayList(child_node),
      new ArrowType.Date(DateUnit.MILLISECOND))
    val funcNode =
      TreeBuilder.makeFunction(
        "extractDay",
        Lists.newArrayList(cast_func),
        new ArrowType.Int(64, true))
    val castNode =
      TreeBuilder.makeFunction("castINT", Lists.newArrayList(funcNode), resultType)
    (castNode, resultType)
  }
}

class ColumnarYear(child: Expression, original: Expression)
  extends Year(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(LongType, StringType, DateType)
    if (supportedTypes.indexOf(child.dataType) == -1) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarYear.")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = new ArrowType.Int(32, true)
    //FIXME(): requires utf8()/int64() as input
    val cast_func =
    TreeBuilder.makeFunction(
      "castDATE",
      Lists.newArrayList(child_node),
      new ArrowType.Date(DateUnit.MILLISECOND))
    val funcNode =
      TreeBuilder.makeFunction(
        "extractYear",
        Lists.newArrayList(cast_func),
        new ArrowType.Int(64, true))
    val castNode =
      TreeBuilder.makeFunction("castINT", Lists.newArrayList(funcNode), resultType)
    (castNode, resultType)
  }
}

class ColumnarNot(child: Expression, original: Expression)
    extends Not(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(BooleanType)
    if (supportedTypes.indexOf(child.dataType) == -1) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarNot.")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = new ArrowType.Bool()
    val funcNode =
      TreeBuilder.makeFunction("not", Lists.newArrayList(child_node), resultType)
    (funcNode, resultType)
  }
}

class ColumnarAbs(child: Expression, original: Expression)
    extends Abs(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(FloatType, DoubleType)
    if (supportedTypes.indexOf(child.dataType) == -1 &&
        !child.dataType.isInstanceOf[DecimalType]) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarAbs")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = CodeGeneration.getResultType(dataType)
    val funcNode =
      TreeBuilder.makeFunction("abs", Lists.newArrayList(child_node), resultType)
    (funcNode, resultType)
  }
}

class ColumnarUpper(child: Expression, original: Expression)
    extends Upper(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(StringType)
    if (supportedTypes.indexOf(child.dataType) == -1) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarUpper")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = new ArrowType.Utf8()
    val funcNode =
      TreeBuilder.makeFunction("upper", Lists.newArrayList(child_node), resultType)
    (funcNode, resultType)
  }
}

class ColumnarBitwiseNot(child: Expression, original: Expression)
    extends BitwiseNot(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(IntegerType, LongType)
    if (supportedTypes.indexOf(child.dataType) == -1) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarBitwiseNot")
    }
  }

  override def doColumnarCodeGen(args: Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val funcNode =
      TreeBuilder.makeFunction("bitwise_not", Lists.newArrayList(child_node), childType)
    (funcNode, childType)
  }
}

class ColumnarCheckOverflow(child: Expression, original: CheckOverflow)
    extends CheckOverflow(
      child: Expression,
      original.dataType: DecimalType,
      original.nullOnOverflow: Boolean)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    if (!child.dataType.isInstanceOf[DecimalType]) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarCheckOverflow")
    }
  }

  override def doColumnarCodeGen(args: Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)
    val newDataType =
      DecimalType(dataType.precision, dataType.scale)
    val resType = CodeGeneration.getResultType(newDataType)
    if (resType == childType) {
      // If target type is the same as childType, cast is not needed
      (child_node, childType)
    } else {
      var function = "castDECIMAL"
      if (nullOnOverflow) {
        function = "castDECIMALNullOnOverflow"
      }
      val funcNode =
        TreeBuilder.makeFunction(function, Lists.newArrayList(child_node), resType)
      (funcNode, resType)
    }
  }
}

class ColumnarCast(
    child: Expression,
    datatype: DataType,
    timeZoneId: Option[String],
    original: Expression)
    extends Cast(child: Expression, datatype: DataType, timeZoneId: Option[String])
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    if (!datatype.isInstanceOf[DecimalType]) {
      try {
        ConverterUtils.checkIfTypeSupported(datatype)
      } catch {
        case e: UnsupportedOperationException =>
          throw new UnsupportedOperationException(s"${datatype} is not supported in ColumnarCast")
      }
      if (datatype == BooleanType) {
        throw new UnsupportedOperationException(s"${datatype} is not supported in ColumnarCast")
      }
    }
    if (datatype == StringType) {
      val supported =
        List(
          ByteType,
          ShortType,
          IntegerType,
          LongType,
          FloatType,
          DoubleType,
          StringType,
          DateType,
          TimestampType)
      if (supported.indexOf(child.dataType) == -1 &&
          !child.dataType.isInstanceOf[DecimalType]) {
        // decimal is supported in castVARCHAR
        throw new UnsupportedOperationException(
          s"${child.dataType} is not supported in castVARCHAR")
      }
    } else if (datatype == ByteType) {
      val supported = List(ShortType, IntegerType, LongType)
      if (supported.indexOf(child.dataType) == -1) {
        throw new UnsupportedOperationException(s"${child.dataType} is not supported in castBYTE")
      }
    } else if (datatype == IntegerType) {
      val supported =
        List(ByteType, ShortType, LongType, FloatType, DoubleType, DateType, DecimalType)
      if (supported.indexOf(child.dataType) == -1 && !child.dataType.isInstanceOf[DecimalType]) {
        throw new UnsupportedOperationException(s"${child.dataType} is not supported in castINT")
      }
    } else if (datatype == LongType) {
      val supported = List(IntegerType, FloatType, DoubleType, DateType, DecimalType)
      if (supported.indexOf(child.dataType) == -1 &&
          !child.dataType.isInstanceOf[DecimalType]) {
        throw new UnsupportedOperationException(
          s"${child.dataType} is not supported in castBIGINT")
      }
    } else if (datatype == FloatType) {
      val supported = List(IntegerType, LongType, DoubleType, DecimalType)
      if (supported.indexOf(child.dataType) == -1 && !child.dataType.isInstanceOf[DecimalType]) {
        throw new UnsupportedOperationException(
          s"${child.dataType} is not supported in castFLOAT4")
      }
    } else if (datatype == DoubleType) {
      val supported = List(IntegerType, LongType, FloatType, DecimalType)
      if (supported.indexOf(child.dataType) == -1 &&
          !child.dataType.isInstanceOf[DecimalType]) {
        throw new UnsupportedOperationException(
          s"${child.dataType} is not supported in castFLOAT8")
      }
    } else if (dataType == DateType) {
      val supported = List(IntegerType, LongType, DateType, TimestampType)
      if (supported.indexOf(child.dataType) == -1) {
        throw new UnsupportedOperationException(s"${child.dataType} is not supported in castDATE")
      }
    } else if (dataType.isInstanceOf[DecimalType]) {
      val supported = List(IntegerType, LongType, FloatType, DoubleType, StringType)
      if (supported.indexOf(child.dataType) == -1 &&
          !child.dataType.isInstanceOf[DecimalType]) {
        throw new UnsupportedOperationException(
          s"${child.dataType} is not supported in castDECIMAL")
      }
    } else if (datatype == TimestampType) {
      val supported = List(LongType)
      if (supported.indexOf(child.dataType) == -1) {
        throw new UnsupportedOperationException(s"${child.dataType} is not supported in castTIMESTAMP")
      }
    } else {
      throw new UnsupportedOperationException(s"not currently supported: ${dataType}.")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = CodeGeneration.getResultType(dataType)
    if (dataType == StringType) {
      val limitLen: java.lang.Long = childType match {
        case int: ArrowType.Int if int.getBitWidth == 8 => 4
        case int: ArrowType.Int if int.getBitWidth == 16 => 6
        case int: ArrowType.Int if int.getBitWidth == 32 => 11
        case int: ArrowType.Int if int.getBitWidth == 64 => 20
        case float: ArrowType.FloatingPoint
            if float.getPrecision() == FloatingPointPrecision.SINGLE =>
          12
        case float: ArrowType.FloatingPoint
            if float.getPrecision() == FloatingPointPrecision.DOUBLE =>
          21
        case date: ArrowType.Date if date.getUnit == DateUnit.DAY => 10
        case decimal: ArrowType.Decimal =>
          // Add two to precision for decimal point and negative sign
          (decimal.getPrecision() + 2)
        case _ =>
          throw new UnsupportedOperationException(
            s"ColumnarCast to String doesn't support ${childType}")
      }
      val limitLenNode = TreeBuilder.makeLiteral(limitLen)
      val funcNode =
        TreeBuilder.makeFunction(
          "castVARCHAR",
          Lists.newArrayList(child_node, limitLenNode),
          resultType)
      (funcNode, resultType)
    } else if (dataType == ByteType) {
      val funcNode =
        TreeBuilder.makeFunction("castBYTE", Lists.newArrayList(child_node), resultType)
      (funcNode, resultType)
    } else if (dataType == IntegerType) {
      val funcNode = child.dataType match {
        case d: DecimalType =>
          val half_node = TreeBuilder.makeDecimalLiteral("0.5", 2, 1)
          val round_down_node = TreeBuilder.makeFunction(
            "subtract",
            Lists.newArrayList(child_node, half_node),
            childType)
          val long_node = TreeBuilder.makeFunction(
            "castBIGINT",
            Lists.newArrayList(round_down_node),
            new ArrowType.Int(64, true))
          TreeBuilder.makeFunction("castINT", Lists.newArrayList(long_node), resultType)
        case other =>
          TreeBuilder.makeFunction("castINT", Lists.newArrayList(child_node), resultType)
      }
      (funcNode, resultType)
    } else if (dataType == LongType) {
      val funcNode =
        TreeBuilder.makeFunction("castBIGINT", Lists.newArrayList(child_node), resultType)
      (funcNode, resultType)
      //(child_node, childType)
    } else if (dataType == FloatType) {
      val funcNode = child.dataType match {
        case d: DecimalType =>
          val double_node = TreeBuilder.makeFunction(
            "castFLOAT8",
            Lists.newArrayList(child_node),
            new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))
          TreeBuilder.makeFunction("castFLOAT4", Lists.newArrayList(double_node), resultType)
        case other =>
          TreeBuilder.makeFunction("castFLOAT4", Lists.newArrayList(child_node), resultType)
      }
      (funcNode, resultType)
    } else if (dataType == DoubleType) {
      val funcNode =
        TreeBuilder.makeFunction("castFLOAT8", Lists.newArrayList(child_node), resultType)
      (funcNode, resultType)
    } else if (dataType == DateType) {
      val castNode = if (childType.isInstanceOf[ArrowType.Timestamp]) {
        val timestampMSType = new ArrowType.Timestamp(TimeUnit.MILLISECOND, CodeGeneration.timeZoneId)
        // cast timestamp(us) => timestamp(ms)
        val timestampMSNode =
          TreeBuilder.makeFunction("convertTimestampUnit", Lists.newArrayList(child_node), timestampMSType)
        // cast timestamp(ms) => date64
        val funcNode =
          TreeBuilder.makeFunction("castDATE", Lists.newArrayList(timestampMSNode), new ArrowType.Date(DateUnit.MILLISECOND))
        // cast date64 => date32
        TreeBuilder.makeFunction("castDATE", Lists.newArrayList(funcNode), resultType)
      } else {
        TreeBuilder.makeFunction("castDATE", Lists.newArrayList(child_node), resultType)
      }

      (castNode, resultType)
    } else if (dataType.isInstanceOf[DecimalType]) {
      dataType match {
        case d: DecimalType =>
          val dType = CodeGeneration.getResultType(d)
          val funcNode =
            TreeBuilder.makeFunction("castDECIMAL", Lists.newArrayList(child_node), dType)
          (funcNode, dType)
      }
    } else if (dataType == TimestampType) {
      val timestampMSType = new ArrowType.Timestamp(TimeUnit.MILLISECOND, CodeGeneration.timeZoneId)
      // cast date64 => timestamp(ms)
      val timestampMSNode =
        TreeBuilder.makeFunction("castTIMESTAMP", Lists.newArrayList(child_node), timestampMSType)
      // cast timestamp(ms) => timestamp(us)
      val funcNode =
        TreeBuilder.makeFunction("convertTimestampUnit", Lists.newArrayList(timestampMSNode), resultType)
      (funcNode, resultType)
    } else {
      throw new UnsupportedOperationException(s"not currently supported: ${dataType}.")
    }
  }
}

class ColumnarUnscaledValue(child: Expression, original: Expression)
    extends UnscaledValue(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    if (!child.dataType.isInstanceOf[DecimalType] && !child.dataType.isInstanceOf[LongType]) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarUnscaledValue")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val resultType = new ArrowType.Int(64, true)
    if (child.dataType.isInstanceOf[LongType]) {
      return (child_node, resultType)
    }
    val childDataType = child.dataType.asInstanceOf[DecimalType]
    val m = ConverterUtils.powerOfTen(childDataType.scale)
    val multiplyType = DecimalTypeUtil.getResultTypeForOperation(
      DecimalTypeUtil.OperationType.MULTIPLY,
      childType.asInstanceOf[ArrowType.Decimal],
      new ArrowType.Decimal((m._2).toInt, (m._3).toInt, 128))
    val increaseScaleNode =
      TreeBuilder.makeFunction(
        "multiply",
        Lists.newArrayList(child_node, TreeBuilder.makeDecimalLiteral(m._1, m._2, m._3)),
        multiplyType)
    val funcNode =
      TreeBuilder.makeFunction("castBIGINT", Lists.newArrayList(increaseScaleNode), resultType)
    (funcNode, resultType)
  }
}

class ColumnarMakeDecimal(
    child: Expression,
    precision: Int,
    scale: Int,
    nullOnOverflow: Boolean,
    original: Expression)
    extends MakeDecimal(child: Expression, precision: Int, scale: Int, nullOnOverflow: Boolean)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    if (!child.dataType.isInstanceOf[LongType]) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarMakeDecimal")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val origType = new ArrowType.Decimal(precision, 0, 128)
    val resultType = new ArrowType.Decimal(precision, scale, 128)
    val m = ConverterUtils.powerOfTen(scale)
    val decimalNode =
      TreeBuilder.makeFunction("castDECIMAL", Lists.newArrayList(child_node), origType)
    val reducedScaleNode =
      TreeBuilder.makeFunction(
        "divide",
        Lists.newArrayList(decimalNode, TreeBuilder.makeDecimalLiteral(m._1, m._2, m._3)),
        resultType)
    (reducedScaleNode, resultType)
  }
}

class ColumnarNormalizeNaNAndZero(child: Expression, original: NormalizeNaNAndZero)
    extends NormalizeNaNAndZero(child: Expression)
    with ColumnarExpression
    with Logging {

  buildCheck()

  def buildCheck(): Unit = {
    val supportedTypes = List(FloatType, DoubleType)
    if (supportedTypes.indexOf(child.dataType) == -1) {
      throw new UnsupportedOperationException(
        s"${child.dataType} is not supported in ColumnarNormalizeNaNAndZero")
    }
  }

  override def doColumnarCodeGen(args: java.lang.Object): (TreeNode, ArrowType) = {
    val (child_node, childType): (TreeNode, ArrowType) =
      child.asInstanceOf[ColumnarExpression].doColumnarCodeGen(args)

    val normalizeNode = TreeBuilder.makeFunction(
      "normalize", Lists.newArrayList(child_node), childType)
    (normalizeNode, childType)
  }
}

object ColumnarUnaryOperator {

  def create(child: Expression, original: Expression): Expression = original match {
    case in: IsNull =>
      new ColumnarIsNull(child, in)
    case i: IsNotNull =>
      new ColumnarIsNotNull(child, i)
    case y: Year =>
      new ColumnarYear(child, y)
    case m: Month =>
      new ColumnarMonth(child, m)
    case d: DayOfMonth =>
      new ColumnarDayOfMonth(child, d)
    case n: Not =>
      new ColumnarNot(child, n)
    case a: Abs =>
      new ColumnarAbs(child, a)
    case u: Upper =>
      new ColumnarUpper(child, u)
    case c: Cast =>
      new ColumnarCast(child, c.dataType, c.timeZoneId, c)
    case u: UnscaledValue =>
      new ColumnarUnscaledValue(child, u)
    case u: MakeDecimal =>
      new ColumnarMakeDecimal(child, u.precision, u.scale, u.nullOnOverflow, u)
    case n: BitwiseNot =>
      new ColumnarBitwiseNot(child, n)
    case a: KnownFloatingPointNormalized =>
      child
    case n: NormalizeNaNAndZero =>
      new ColumnarNormalizeNaNAndZero(child, n)
    case a: PromotePrecision =>
      child
    case a: CheckOverflow =>
      new ColumnarCheckOverflow(child, a)
    case other =>
      throw new UnsupportedOperationException(s"not currently supported: $other.")
  }
}
