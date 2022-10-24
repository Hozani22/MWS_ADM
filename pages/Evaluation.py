from pages.KNN import dKNN
from pages.bayes import dbayes
from pages.DecisionTree import dDecisionTree
from pages.logistic_regression import lr_prediction2
import pandas as pd
import streamlit as st


All1 = { '' : dDecisionTree}
All2 = { ' ': dbayes}
All3 = { ' ': dKNN}
All4 = { ' ': lr_prediction2}

st.write(pd.DataFrame(data=All1))
st.write(pd.DataFrame(data=All2))
st.write(pd.DataFrame(data=All3))
st.write(pd.DataFrame(data=All4))