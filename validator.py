from wtforms import Form, StringField, IntegerField, FloatField
from wtforms.validators import InputRequired, Length, DataRequired
from flask_inputs import Inputs


class RsForm(Form):

    Tool = StringField('Tool', validators=[InputRequired(),DataRequired(),
                                           Length(min=1, max=100)])
    Rs_Plan = StringField('Rs_Plan', validators=[InputRequired(),
                                                 Length(min=1, max=100)])
    Unit = StringField('Unit', validators=[InputRequired(),
                                           Length(min=1, max=100)])
    Language_Pair = StringField('Language_Pair', validators=[InputRequired(),
                                                             Length(min=1, max=191)])
    Subject = StringField('Subject', validators=[InputRequired(),
                                                 Length(min=1, max=191)])
    Brand = StringField('Brand', validators=[InputRequired(),
                                             Length(min=1, max=191)])
    Task_Type = StringField('Task_Type', validators=[InputRequired(),
                                                     Length(min=1, max=191)])

    hour = IntegerField('hour', validators=[InputRequired()])
    rs_m_day = IntegerField('rs_m_day', validators=[InputRequired()])
    rs_month = IntegerField('rs_month', validators=[InputRequired()])
    duration = FloatField('duration', validators=[InputRequired()])
    dateStamp_base = FloatField('dateStamp_base', validators=[InputRequired()])
    unified_task_amount = FloatField('unified_task_amount', validators=[InputRequired()])


class RSInputs(Inputs):
    rule = {
        'Tool': [DataRequired()],
        'Rs_Plan': [DataRequired()],
        'Unit': [DataRequired()],
        'Language_Pair': [DataRequired()],
        'Subject': [DataRequired()],
        'Brand': [DataRequired()],
        'Task_Type': [DataRequired()],
        'hour': [DataRequired()],
        'rs_m_day': [DataRequired()],
        'rs_month': [DataRequired()],
        'duration': [DataRequired()],
        'dateStamp_base': [DataRequired()]
    }
