# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:05:08 2021

@author: jamis

一秒おきにコマンドプロンプトに対して"1"を入力します
"""
import subprocess
import time

a = 0
while True:
    
    subprocess.run([str(1)],shell=True)
    time.sleep(1)
    
    a += 1
    
    if a == 20:
        break



