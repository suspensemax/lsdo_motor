__version__ = '0.0.1'

try:
    from lsdo_motor.core.old.motor_analysis_m3l_model import MotorAnalysis, evaluate_multiple_motor_analysis_models
    from lsdo_motor.core.old.motor_sizing_m3l_model import MotorSizing, evaluate_multiple_motor_sizing_models
except:
    pass

from lsdo_motor.core.old.TC1_motor_analysis_model import TC1MotorAnalysisModel
from lsdo_motor.core.old.TC1_motor_sizing_model import TC1MotorSizingModel
from lsdo_motor.core.old.TC1_full_motor_model import FullMotorModel

from lsdo_motor.core.motor_model import MotorModel, MotorInputVariableGroup

from pathlib import Path
ROOT = Path(__file__).parents[0]
GEOMETRY_PATH = ROOT / 'core' / 'sample_geometry' / 'geometry'
IMPORTS_PATH = ROOT / 'core' / 'sample_geometry' / 'imports'