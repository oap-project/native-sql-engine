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

package com.intel.oap.vectorized;

import org.apache.spark.sql.catalyst.expressions.UnsafeRow;

import java.io.IOException;

public class ArrowColumnarToRowJniWrapper {

  public ArrowColumnarToRowJniWrapper() throws IOException {
    JniUtils.getInstance();
  }

  public native long nativeConvertColumnarToRow(
    byte[] schema, int numRows, long[] bufAddrs, long[] bufSizes, long memory_pool_id) throws RuntimeException;

  public native boolean nativeHasNext(long instanceID);

  public native UnsafeRow nativeNext(long instanceID);
}
