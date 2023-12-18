#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, May 2nd 2023, 3:25:40 pm
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose", "template_name")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        self.template_name = template_name

        file_name = osp.join("src", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )


    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)

        return res

    def get_response(self, output: str) -> str:

        if self.template["response_split"] not in output:
            return output

        if len(output.split(self.template["response_split"])) < 2:
            print("Empty response: " + output)
            return "EMPTY RESPONSE"
        else:
            return output.split(self.template["response_split"])[1].strip()