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

package org.apache.spark.sql.execution

import java.util.concurrent.atomic.AtomicInteger

import com.intel.oap.execution._
import com.intel.oap.expression.ColumnarExpressionConverter
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.execution._
import org.apache.spark.sql.catalyst.plans._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.sql.types.ObjectType

/**
 * InputAdapter is used to hide a SparkPlan from a subtree that supports codegen.
 */
class ColumnarInputAdapter(child: SparkPlan) extends InputAdapter(child) {

  // This is not strictly needed because the codegen transformation happens after the columnar
  // transformation but just for consistency
  override def supportsColumnar: Boolean = child.supportsColumnar
  override def supportCodegen: Boolean = false

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    child.executeColumnar()
  }

  override def nodeName: String = s"InputAdapter"

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
    child.generateTreeString(
      depth,
      lastChildren,
      append,
      verbose,
      prefix = "",
      addSuffix = false,
      maxFields,
      printNodeId,
      indent)
  }
}

/**
 * Find the chained plans that support codegen, collapse them together as WholeStageCodegen.
 *
 * The `codegenStageCounter` generates ID for codegen stages within a query plan.
 * It does not affect equality, nor does it participate in destructuring pattern matching
 * of WholeStageCodegenExec.
 *
 * This ID is used to help differentiate between codegen stages. It is included as a part
 * of the explain output for physical plans, e.g.
 *
 * == Physical Plan ==
 * *(5) SortMergeJoin [x#3L], [y#9L], Inner
 * :- *(2) Sort [x#3L ASC NULLS FIRST], false, 0
 * :  +- Exchange hashpartitioning(x#3L, 200)
 * :     +- *(1) Project [(id#0L % 2) AS x#3L]
 * :        +- *(1) Filter isnotnull((id#0L % 2))
 * :           +- *(1) Range (0, 5, step=1, splits=8)
 * +- *(4) Sort [y#9L ASC NULLS FIRST], false, 0
 *    +- Exchange hashpartitioning(y#9L, 200)
 *       +- *(3) Project [(id#6L % 2) AS y#9L]
 *          +- *(3) Filter isnotnull((id#6L % 2))
 *             +- *(3) Range (0, 5, step=1, splits=8)
 *
 * where the ID makes it obvious that not all adjacent codegen'd plan operators are of the
 * same codegen stage.
 *
 * The codegen stage ID is also optionally included in the name of the generated classes as
 * a suffix, so that it's easier to associate a generated class back to the physical operator.
 * This is controlled by SQLConf: spark.sql.codegen.useIdInClassName
 *
 * The ID is also included in various log messages.
 *
 * Within a query, a codegen stage in a plan starts counting from 1, in "insertion order".
 * WholeStageCodegenExec operators are inserted into a plan in depth-first post-order.
 * See CollapseCodegenStages.insertWholeStageCodegen for the definition of insertion order.
 *
 * 0 is reserved as a special ID value to indicate a temporary WholeStageCodegenExec object
 * is created, e.g. for special fallback handling when an existing WholeStageCodegenExec
 * failed to generate/compile code.
 */
case class ColumnarCollapseCodegenStages(
    columnarWholeStageEnabled: Boolean,
    codegenStageCounter: AtomicInteger = new AtomicInteger(0))
    extends Rule[SparkPlan] {

  private def supportCodegen(plan: SparkPlan): Boolean = plan match {
    case plan: ColumnarCodegenSupport =>
      plan.supportColumnarCodegen
    case _ => false
  }

  private def containsSubquery(expr: Expression): Boolean = {
    if (expr == null) false
    else {
      ColumnarExpressionConverter.containsSubquery(expr)
    }
  }

  private def containsSubquery(exprs: Seq[Expression]): Boolean = {
    if (exprs == null) false
    else {
      exprs.map(ColumnarExpressionConverter.containsSubquery).exists(_ == true)
    }
  }

  private def existsJoins(plan: SparkPlan, count: Int = 0): Boolean = plan match {
    case p: ColumnarBroadcastHashJoinExec =>
      if (p.condition.isDefined) return true
      if (count >= 1) true
      else plan.children.map(existsJoins(_, count + 1)).exists(_ == true)
    case p: ColumnarShuffledHashJoinExec =>
      if (p.condition.isDefined) return true
      if (count >= 1) true
      else plan.children.map(existsJoins(_, count + 1)).exists(_ == true)
    case p: ColumnarSortMergeJoinExec =>
      true
    case p: ColumnarHashAggregateExec =>
      if (count >= 1) true
      else plan.children.map(existsJoins(_, count + 1)).exists(_ == true)
    case p: ColumnarConditionProjectExec
        if (containsSubquery(p.condition) || containsSubquery(p.projectList)) =>
      false
    case p: ColumnarCodegenSupport if p.supportColumnarCodegen =>
      plan.children.map(existsJoins(_, count)).exists(_ == true)
    case _ =>
      false
  }

  private def containsExpression(expr: Expression): Boolean = expr match {
    case a: Alias =>
      containsExpression(a.child)
    case a: AttributeReference =>
      false
    case other =>
      true
  }

  private def containsExpression(projectList: Seq[NamedExpression]): Boolean = {
    projectList.map(containsExpression).exists(_ == true)
  }

  private def joinOptimization(
      plan: ColumnarConditionProjectExec,
      skip_smj: Boolean = false): SparkPlan = plan.child match {
    case p: ColumnarBroadcastHashJoinExec
        if plan.condition == null && !containsExpression(plan.projectList) =>
      ColumnarBroadcastHashJoinExec(
        p.leftKeys,
        p.rightKeys,
        p.joinType,
        p.buildSide,
        p.condition,
        p.left,
        p.right,
        plan.projectList,
        nullAware = p.isNullAwareAntiJoin)
    case p: ColumnarShuffledHashJoinExec
        if plan.condition == null && !containsExpression(plan.projectList) =>
      ColumnarShuffledHashJoinExec(
        p.leftKeys,
        p.rightKeys,
        p.joinType,
        p.buildSide,
        p.condition,
        p.left,
        p.right,
        plan.projectList)
    case p: ColumnarSortMergeJoinExec
        if !skip_smj && plan.condition == null && !containsExpression(plan.projectList) =>
      ColumnarSortMergeJoinExec(
        p.leftKeys,
        p.rightKeys,
        p.joinType,
        p.condition,
        p.left,
        p.right,
        p.isSkewJoin,
        plan.projectList)
    case other => plan
  }

  /**
   * Inserts an InputAdapter on top of those that do not support codegen.
   */
  private def insertInputAdapter(plan: SparkPlan): SparkPlan = {
    plan match {
      case p if !supportCodegen(p) =>
        new ColumnarInputAdapter(insertWholeStageCodegen(p))
      case p: ColumnarConditionProjectExec
          if (containsSubquery(p.condition) || containsSubquery(p.projectList)) =>
        new ColumnarInputAdapter(p.withNewChildren(p.children.map(insertWholeStageCodegen)))
      case j: ColumnarSortMergeJoinExec
          if j.buildPlan.isInstanceOf[ColumnarSortMergeJoinExec] || (j.buildPlan
            .isInstanceOf[ColumnarConditionProjectExec] && j.buildPlan
            .children(0)
            .isInstanceOf[ColumnarSortMergeJoinExec]) =>
        // we don't support any ColumnarSortMergeJoin whose both children are ColumnarSortMergeJoin
        j.withNewChildren(j.children.map(c => {
          if (c.equals(j.buildPlan)) {
            new ColumnarInputAdapter(insertWholeStageCodegen(c))
          } else {
            insertInputAdapter(c)
          }
        }))
      case j: ColumnarHashAggregateExec =>
        new ColumnarInputAdapter(insertWholeStageCodegen(j))
      case j: ColumnarSortExec =>
        j.withNewChildren(
          j.children.map(child => new ColumnarInputAdapter(insertWholeStageCodegen(child))))
      case p =>
        p match {
          case exec: ColumnarConditionProjectExec =>
            val after_opt = joinOptimization(exec)
            if (after_opt.isInstanceOf[ColumnarConditionProjectExec]) {
              after_opt.withNewChildren(after_opt.children.map(c => {
                if (c.isInstanceOf[ColumnarSortExec]) {
                  new ColumnarInputAdapter(insertWholeStageCodegen(c))
                } else {
                  insertInputAdapter(c)
                }
              }))
            } else {
              after_opt.withNewChildren(after_opt.children.map(c => {
                insertInputAdapter(c)
              }))
            }
          case _ =>
            p.withNewChildren(p.children.map(insertInputAdapter))
        }
    }
  }

  /**
   * Inserts a WholeStageCodegen on top of those that support codegen.
   */
  private def insertWholeStageCodegen(plan: SparkPlan): SparkPlan = {
    plan match {
      // For operators that will output domain object, do not insert WholeStageCodegen for it as
      // domain object can not be written into unsafe row.
      case plan
          if plan.output.length == 1 && plan.output.head.dataType.isInstanceOf[ObjectType] =>
        plan.withNewChildren(plan.children.map(insertWholeStageCodegen))
      case j: ColumnarHashAggregateExec =>
        if (!j.child.isInstanceOf[ColumnarHashAggregateExec] && existsJoins(j)) {
          ColumnarWholeStageCodegenExec(j.withNewChildren(j.children.map(insertInputAdapter)))(
            codegenStageCounter.incrementAndGet())
        } else {
          j.withNewChildren(j.children.map(insertWholeStageCodegen))
        }
      case s: ColumnarSortExec =>
        /*If ColumnarSort is not ahead of ColumnarSMJ, we should not do wscg for it*/
        s.withNewChildren(s.children.map(insertWholeStageCodegen))
      case plan: ColumnarCodegenSupport if supportCodegen(plan) && existsJoins(plan) =>
        ColumnarWholeStageCodegenExec(insertInputAdapter(plan))(
          codegenStageCounter.incrementAndGet())
      case other =>
        if (other.isInstanceOf[ColumnarConditionProjectExec]) {
          val after_opt =
            joinOptimization(other.asInstanceOf[ColumnarConditionProjectExec], skip_smj = true)
          after_opt.withNewChildren(after_opt.children.map(insertWholeStageCodegen))
        } else {
          other.withNewChildren(other.children.map(insertWholeStageCodegen))
        }
    }
  }

  def apply(plan: SparkPlan): SparkPlan = {
    if (columnarWholeStageEnabled) {
      insertWholeStageCodegen(plan)
    } else {
      plan
    }
  }
}
