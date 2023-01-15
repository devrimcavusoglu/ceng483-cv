#!/bin/sh
set -e

tar --exclude='checkpoints' --exclude='ceng483-f22-hw3-dataset' --exclude='*.pdf' --exclude="*.sh" -czf 2010023_the3.tar.gz .
sleep 2