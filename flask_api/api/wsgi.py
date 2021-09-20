# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 16:52:25 2021

@author: ouss3ma
"""

from .app import app

if __name__ == "__main__":
    app.run(use_reload=True, debug=True)
    