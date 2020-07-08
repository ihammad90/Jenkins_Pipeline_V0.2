import pytest
import math
from training import train_function



def test_r2_score_train():

	subject = train_function()
	assert math.ceil(subject)==1
