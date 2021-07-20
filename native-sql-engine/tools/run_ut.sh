#!/bin/bash

# This script is used to run native sql unit test
# SPARK_HOME is required, Usage: ./run_ut.sh
# Detailed test info is logged to oap-native-sql/tools/log-file.log

cd ../..
spark_home=$(eval echo ${SPARK_HOME})
if [ -z "${spark_home}" ]
then
  echo "SPARK_HOME is not set!"
  exit 1
else
  echo "SPARK_HOME is $spark_home"
fi
count=0; while [ $count -le 40 ]; do echo "Elapsed 3 minutes"; sleep 180; let count++; done & # print log
mvn clean test -P full-scala-compiler -am -pl native-sql-engine/core -Dbuild_arrow=OFF -Dbuild_protobuf=OFF -DfailIfNoTests=false -DargLine="-Dspark.test.home=$spark_home" -Dexec.skip=true -Dmaven.test.failure.ignore=true
# &>  native-sql-engine/tools/log-file.log
cd native-sql-engine/tools/
tests_total=0
module_tested=0
module_should_test=1
while read -r line ; do
  num=$(echo "$line" | grep -o -E '[0-9]+')
  tests_total=$((tests_total+num))
done <<<"$(grep "Total number of tests run:" log-file.log)"
 
succeed_total=0
while read -r line ; do
  [[ $line =~ [^0-9]*([0-9]+)\, ]]
  num=${BASH_REMATCH[1]}
  succeed_total=$((succeed_total+num))
  let module_tested++
done <<<"$(grep "succeeded" log-file.log)"
echo "Tests total: $tests_total, Succeed Total: $succeed_total"
 
if test $tests_total -eq $succeed_total -a $module_tested -eq $module_should_test
then
  echo "All unit tests succeed"
else
  echo "Unit tests failed, please check log-file.log for detailed info"
  exit 1
fi
