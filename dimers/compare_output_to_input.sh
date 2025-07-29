#!/bin/bash

awk '/Variables initialized/{flag=1; next} /Commands initialized/{flag=0} flag' $1 > __input.tmp

sed -i 's/^ //g' __input.tmp

diff __input.tmp $2

