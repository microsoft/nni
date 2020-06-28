#!/bin/bash
set -e
CWD=${PWD}

## Export certain environment variables for unittest code to work
export COVERAGE_PROCESS_START=${CWD}/.coveragerc
export COVERAGE_DATA_FILE=${CWD}/coverage/data
export COVERAGE_HTML_DIR=${CWD}/coverhtml

rm ${COVERAGE_DATA_FILE}*
rm -rf ${COVERAGE_HTML_DIR}/*
mkdir ${CWD}/coverage
mkdir ${COVERAGE_HTML_DIR}

## ------Run integration test------
echo "===========================Testing: integration test==========================="
coverage run sdk_test.py
coverage combine
coverage html
