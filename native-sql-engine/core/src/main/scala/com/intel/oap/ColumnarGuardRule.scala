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

package com.intel.oap

import com.intel.oap.execution._
import org.apache.spark.SparkConf
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, SparkSessionExtensions}
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.catalyst.plans.FullOuter
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.adaptive._
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec
import org.apache.spark.sql.execution.exchange._
import org.apache.spark.sql.execution.joins._
import org.apache.spark.sql.execution.window.WindowExec
import org.apache.spark.sql.internal.SQLConf

case class RowGuard(child: SparkPlan) extends SparkPlan {
  def output: Seq[Attribute] = child.output
  protected def doExecute(): RDD[InternalRow] = {
    throw new UnsupportedOperationException
  }
  def children: Seq[SparkPlan] = Seq(child)
}

case class ColumnarGuardRule(conf: SparkConf) extends Rule[SparkPlan] {
  val columnarConf = ColumnarPluginConfig.getSessionConf
  val preferColumnar = columnarConf.enablePreferColumnar
  val optimizeLevel = columnarConf.joinOptimizationThrottle
  val enableColumnarShuffle = columnarConf.enableColumnarShuffle
  val enableColumnarSort = columnarConf.enableColumnarSort
  val enableColumnarWindow = columnarConf.enableColumnarWindow
  val enableColumnarSortMergeJoin = columnarConf.enableColumnarSortMergeJoin
  val testing = columnarConf.isTesting

  private def tryConvertToColumnar(plan: SparkPlan): Boolean = {
    try {
      val columnarPlan = plan match {
        case plan: BatchScanExec =>
          if (testing) {
            // disable ColumnarBatchScanExec according to config
            return false
          }
          new ColumnarBatchScanExec(plan.output, plan.scan)
        case plan: FileSourceScanExec =>
          if (plan.supportsColumnar) {
            return false
          }
          plan
        case plan: InMemoryTableScanExec =>
          new ColumnarInMemoryTableScanExec(plan.attributes, plan.predicates, plan.relation)
        case plan: ProjectExec =>
          new ColumnarConditionProjectExec(null, plan.projectList, plan.child)
        case plan: FilterExec =>
          new ColumnarConditionProjectExec(plan.condition, null, plan.child)
        case plan: HashAggregateExec =>
          new ColumnarHashAggregateExec(
            plan.requiredChildDistributionExpressions,
            plan.groupingExpressions,
            plan.aggregateExpressions,
            plan.aggregateAttributes,
            plan.initialInputBufferOffset,
            plan.resultExpressions,
            plan.child)
        case plan: UnionExec =>
          new ColumnarUnionExec(plan.children)
        case plan: ExpandExec =>
          new ColumnarExpandExec(plan.projections, plan.output, plan.child)
        case plan: SortExec =>
          if (!enableColumnarSort) return false
          new ColumnarSortExec(plan.sortOrder, plan.global, plan.child, plan.testSpillFrequency)
        case plan: ShuffleExchangeExec =>
          if (!enableColumnarShuffle) return false
          new ColumnarShuffleExchangeExec(
            plan.outputPartitioning,
            plan.child,
            plan.canChangeNumPartitions)
        case plan: ShuffledHashJoinExec =>
          ColumnarShuffledHashJoinExec(
            plan.leftKeys,
            plan.rightKeys,
            plan.joinType,
            plan.buildSide,
            plan.condition,
            plan.left,
            plan.right)
        case plan: BroadcastExchangeExec =>
          ColumnarBroadcastExchangeExec(plan.mode, plan.child)
        case plan: BroadcastHashJoinExec =>
          // We need to check if BroadcastExchangeExec can be converted to columnar-based.
          // If not, BHJ should also be row-based.
          val left = plan.left
          left match {
            case exec: BroadcastExchangeExec =>
              new ColumnarBroadcastExchangeExec(exec.mode, exec.child)
            case BroadcastQueryStageExec(_, plan: BroadcastExchangeExec) =>
              new ColumnarBroadcastExchangeExec(plan.mode, plan.child)
            case BroadcastQueryStageExec(_, plan: ReusedExchangeExec) =>
              plan match {
                case ReusedExchangeExec(_, b: BroadcastExchangeExec) =>
                  new ColumnarBroadcastExchangeExec(b.mode, b.child)
                case _ =>
              }
            case _ =>
          }
          val right = plan.right
          right match {
            case exec: BroadcastExchangeExec =>
              new ColumnarBroadcastExchangeExec(exec.mode, exec.child)
            case BroadcastQueryStageExec(_, plan: BroadcastExchangeExec) =>
              new ColumnarBroadcastExchangeExec(plan.mode, plan.child)
            case BroadcastQueryStageExec(_, plan: ReusedExchangeExec) =>
              plan match {
                case ReusedExchangeExec(_, b: BroadcastExchangeExec) =>
                  new ColumnarBroadcastExchangeExec(b.mode, b.child)
                case _ =>
              }
            case _ =>
          }
          ColumnarBroadcastHashJoinExec(
            plan.leftKeys,
            plan.rightKeys,
            plan.joinType,
            plan.buildSide,
            plan.condition,
            plan.left,
            plan.right)
        case plan: SortMergeJoinExec =>
          if (!enableColumnarSortMergeJoin || plan.joinType == FullOuter) return false
          new ColumnarSortMergeJoinExec(
            plan.leftKeys,
            plan.rightKeys,
            plan.joinType,
            plan.condition,
            plan.left,
            plan.right,
            plan.isSkewJoin)
        case plan: WindowExec =>
          if (!enableColumnarWindow) return false
          val window = ColumnarWindowExec.create(
            plan.windowExpression,
            plan.partitionSpec,
            plan.orderSpec,
            plan.child)
          window
        case p =>
          p
      }
    } catch {
      case e: UnsupportedOperationException =>
        plan match {
          case plan: HashAggregateExec =>
            val queryInfo =
              s"HashAggr groupingExpressions is ${plan.groupingExpressions.toList.map(g =>
                (g.dataType, g))}\naggregateExpressions is ${plan.aggregateExpressions.toList
                .map(a => (a.dataType, a))}\nresultExpressions is ${plan.resultExpressions.toList
                .map(e => (e.dataType, e))}"
            System.out.println(queryInfo)
          case other => {}
        }
        System.out.println(
          s"Fall back to use row-based operators, error is ${e.getMessage}, original sparkplan is ${plan.getClass}(${plan.children.toList
            .map(_.getClass)})")
        return false
    }
    return true
  }

  private def existsMultiCodegens(plan: SparkPlan, count: Int = 0): Boolean =
    plan match {
      case plan: CodegenSupport if plan.supportCodegen =>
        if ((count + 1) >= optimizeLevel) return true
        plan.children.map(existsMultiCodegens(_, count + 1)).exists(_ == true)
      case plan: ShuffledHashJoinExec =>
        if ((count + 1) >= optimizeLevel) return true
        plan.children.map(existsMultiCodegens(_, count + 1)).exists(_ == true)
      case other => false
    }

  private def supportCodegen(plan: SparkPlan): Boolean = plan match {
    case plan: CodegenSupport =>
      plan.supportCodegen
    case _ => false
  }

  /**
   * Inserts an InputAdapter on top of those that do not support codegen.
   */
  private def insertRowGuardRecursive(plan: SparkPlan): SparkPlan = {
    plan match {
      case p: ShuffleExchangeExec =>
        RowGuard(p.withNewChildren(p.children.map(insertRowGuardOrNot)))
      case p: BroadcastExchangeExec =>
        RowGuard(p.withNewChildren(p.children.map(insertRowGuardOrNot)))
      case p: ShuffledHashJoinExec =>
        RowGuard(p.withNewChildren(p.children.map(insertRowGuardRecursive)))
      case p if !supportCodegen(p) =>
        // insert row guard them recursively
        p.withNewChildren(p.children.map(insertRowGuardOrNot))
      case p: CustomShuffleReaderExec =>
        p.withNewChildren(p.children.map(insertRowGuardOrNot))
      case p: BroadcastQueryStageExec =>
        p
      case p => RowGuard(p.withNewChildren(p.children.map(insertRowGuardRecursive)))
    }
  }

  private def insertRowGuard(plan: SparkPlan): SparkPlan = {
    RowGuard(plan.withNewChildren(plan.children.map(insertRowGuardOrNot)))
  }

  /**
   * Inserts a WholeStageCodegen on top of those that support codegen.
   */
  private def insertRowGuardOrNot(plan: SparkPlan): SparkPlan = {
    plan match {
      // For operators that will output domain object, do not insert WholeStageCodegen for it as
      // domain object can not be written into unsafe row.
      case plan if !preferColumnar && existsMultiCodegens(plan) =>
        insertRowGuardRecursive(plan)
      case plan if !tryConvertToColumnar(plan) =>
        insertRowGuard(plan)
      case p: BroadcastQueryStageExec =>
        p
      case other =>
        other.withNewChildren(other.children.map(insertRowGuardOrNot))
    }
  }

  def apply(plan: SparkPlan): SparkPlan = {
    insertRowGuardOrNot(plan)
  }
}
