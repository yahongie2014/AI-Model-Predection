from wtforms import Form, StringField, IntegerField, FloatField
from wtforms.validators import InputRequired, Length, DataRequired
from flask_inputs import Inputs


class RsForm(Form):
    Tool = StringField('Tool', validators=[InputRequired(), DataRequired(),
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


class ScoreForm(Form):
    Vendor_types_name = StringField('Vendor_types_name', validators=[InputRequired(), DataRequired(),
                                                                     Length(min=1, max=100)])
    total_gross_amount_Main_currency = FloatField('total_gross_amount_Main_currency', validators=[InputRequired()])
    diff_Due_task = IntegerField('diff_Due_task', validators=[InputRequired()])
    Count_Tasks = IntegerField('Count_Tasks', validators=[InputRequired()])
    cost_day = FloatField('cost_day', validators=[InputRequired()])
    task_amount_days = FloatField('task_amount_days', validators=[InputRequired()])


class ProfitabilityForm(Form):
    Brand = StringField('Brand', validators=[InputRequired(), DataRequired(),
                                             Length(min=1, max=100)])
    Unit = StringField('Unit', validators=[InputRequired(),
                                           Length(min=1, max=100)])
    Job_type = StringField('Job_type', validators=[InputRequired(),
                                                   Length(min=1, max=100)])
    Subject = StringField('Subject', validators=[InputRequired(),
                                                 Length(min=1, max=191)])
    Language_Pair = StringField('Language_Pair', validators=[InputRequired(),
                                                             Length(min=1, max=191)])
    PM = StringField('PM', validators=[InputRequired(),
                                       Length(min=1, max=191)])
    Account = StringField('Account', validators=[InputRequired(),
                                                 Length(min=1, max=191)])

    Start_TimeStamp = FloatField('Start_TimeStamp', validators=[InputRequired()])
    Price = FloatField('Price', validators=[InputRequired()])
    Deivery_TimeStamp = FloatField('Deivery_TimeStamp', validators=[InputRequired()])
    amount = FloatField('amount', validators=[InputRequired()])
    Duration = FloatField('Duration', validators=[InputRequired()])


class FeedForm(Form):
    Brand = StringField('Brand', validators=[InputRequired(), DataRequired(),
                                             Length(min=1, max=100)])
    Unit = StringField('Unit', validators=[InputRequired(),
                                           Length(min=1, max=100)])
    Job_type = StringField('Job_type', validators=[InputRequired(),
                                                   Length(min=1, max=100)])
    Subject = StringField('Subject', validators=[InputRequired(),
                                                 Length(min=1, max=191)])
    Delay = StringField('Delay', validators=[InputRequired()])

    Language_Pair = StringField('Language_Pair', validators=[InputRequired(),
                                                             Length(min=1, max=191)])
    PM = StringField('PM', validators=[InputRequired(),
                                       Length(min=1, max=191)])
    Account = StringField('Account', validators=[InputRequired(),
                                                 Length(min=1, max=191)])

    Start_TimeStamp = FloatField('Start_TimeStamp', validators=[InputRequired()])
    Price = FloatField('Price', validators=[InputRequired()])
    Deivery_TimeStamp = FloatField('Deivery_TimeStamp', validators=[InputRequired()])
    amount = FloatField('amount', validators=[InputRequired()])
    Duration = FloatField('Duration', validators=[InputRequired()])


class DelayForm(Form):
    Brand = StringField('Brand', validators=[InputRequired(), DataRequired(),
                                             Length(min=1, max=100)])
    Unit = StringField('Unit', validators=[InputRequired(),
                                           Length(min=1, max=100)])
    Job_type = StringField('Job_type', validators=[InputRequired(),
                                                   Length(min=1, max=100)])
    Subject = StringField('Subject', validators=[InputRequired(),
                                                 Length(min=1, max=191)])
    Language_Pair = StringField('Language_Pair', validators=[InputRequired(),
                                                             Length(min=1, max=191)])
    PM = StringField('PM', validators=[InputRequired(),
                                       Length(min=1, max=191)])
    Account = StringField('Account', validators=[InputRequired(),
                                                 Length(min=1, max=191)])

    Start_TimeStamp = FloatField('Start_TimeStamp', validators=[InputRequired()])
    Deivery_TimeStamp = FloatField('Deivery_TimeStamp', validators=[InputRequired()])
    amount = FloatField('amount', validators=[InputRequired()])
    Duration = FloatField('Duration', validators=[InputRequired()])


class PayoutForm(Form):
    account_id = StringField('account_id', validators=[InputRequired(), DataRequired()])
    issue_month = StringField('issue_month', validators=[InputRequired()])
    issue_day = StringField('issue_day', validators=[InputRequired()])
    due_month = StringField('due_month', validators=[InputRequired()])
    due_day = StringField('due_day', validators=[InputRequired()])
    payment_terms = StringField('payment_terms', validators=[InputRequired()])
    credit_history = StringField('credit_history', validators=[InputRequired()])
    paid = StringField('paid', validators=[InputRequired()])
    invoice_amount_main_currency = FloatField('invoice_amount_main_currency', validators=[InputRequired()])


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
