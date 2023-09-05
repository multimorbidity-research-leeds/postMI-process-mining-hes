# -*- coding: utf-8 -*-
"""
@author: Chris Hayward
"""



# class Test_remove_entire_subjects_diag(unittest.TestCase):

#     def setUp(self):
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
#                                 'DIAG_01': ['A00','B50','C99',
#                                             'A00','B50','C99',
#                                             'A00','B50','C99'],
#                                })  
#     # #########
#     # Remove based on DIAG_01
#     # #########
#     def test_remove_subjects_with_invalid_or_empty_diag_01_no_change(self):
#         df_out = clean_hes.remove_subjects_with_invalid_or_empty_diag_01(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)

#     # ###
#     # R69
#     # ###
#     def test_remove_subjects_with_invalid_or_empty_diag_01_R69(self):
#         self.df.loc[0,'DIAG_01'] = 'R69'
#         df_out = clean_hes.remove_subjects_with_invalid_or_empty_diag_01(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)

#     def test_remove_subjects_with_invalid_or_empty_diag_01_R69_all(self):
#         self.df.loc[0,'DIAG_01'] = 'R69X'
#         self.df.loc[3,'DIAG_01'] = 'R69A'
#         self.df.loc[6,'DIAG_01'] = 'R69BB'
#         df_out = clean_hes.remove_subjects_with_invalid_or_empty_diag_01(self.df.copy())
        
#         df_empty = self.df.drop(range(9))
#         df_empty.set_index(np.array([],dtype=object),inplace=True)
        
#         pd.testing.assert_frame_equal(df_empty, df_out)

#     # ###
#     # nan
#     # ###
#     def test_remove_subjects_with_invalid_or_empty_diag_01_nan(self):
#         self.df.loc[0,'DIAG_01'] = np.nan
#         df_out = clean_hes.remove_subjects_with_invalid_or_empty_diag_01(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)

#     def test_remove_subjects_with_invalid_or_empty_diag_01_nan_all(self):
#         self.df.loc[0,'DIAG_01'] = np.nan
#         self.df.loc[3,'DIAG_01'] = np.nan
#         self.df.loc[6,'DIAG_01'] = np.nan
#         df_out = clean_hes.remove_subjects_with_invalid_or_empty_diag_01(self.df.copy())
        
#         df_empty = self.df.drop(range(9))
#         df_empty.set_index(np.array([],dtype=object),inplace=True)
        
#         pd.testing.assert_frame_equal(df_empty, df_out)


        
# class Test_remove_subjects_nan_epistart(unittest.TestCase):
    
#     def setUp(self):
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
#                                 'MYEPISTART':np.arange('2005-01',
#                                                        '2005-10', dtype='datetime64[M]')
#                                })
    
#     def test_remove_subjects_nan_epistart_ok(self):        
#         df_out = clean_hes.remove_subjects_nan_epistart(self.df.copy())
#         df_expected = self.df.copy()
#         pd.testing.assert_frame_equal(df_expected, df_out)
        
#     def test_remove_subjects_nan_epistart_remove(self):
#         self.df.loc[0, 'MYEPISTART'] = np.nan
#         df_out = clean_hes.remove_subjects_nan_epistart(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].copy()
#         pd.testing.assert_frame_equal(df_expected.reset_index(drop=True), df_out)
        
        
# class Test_remove_subjects_nan_epiend(unittest.TestCase):
    
#     def setUp(self):
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
#                                 'MYEPIEND':np.arange('2005-01',
#                                                      '2005-10', dtype='datetime64[M]')
#                                })
    
#     def test_remove_subjects_nan_epiend_ok(self):        
#         df_out = clean_hes.remove_subjects_nan_epiend(self.df.copy())
#         df_expected = self.df.copy()
#         pd.testing.assert_frame_equal(df_expected, df_out)
        
#     def test_remove_subjects_nan_epiend_remove(self):
#         self.df.loc[0, 'MYEPIEND'] = np.nan
#         df_out = clean_hes.remove_subjects_nan_epiend(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].copy()
#         pd.testing.assert_frame_equal(df_expected.reset_index(drop=True), df_out)
        
        


# class Test_remove_subjects_disdate_before_admidate(unittest.TestCase):
    
#     def setUp(self):
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
#                                 'MYEPISTART':
#                                     np.concatenate(
#                                           [np.datetime64('2005-01-01'),
#                                           np.datetime64('2005-02-01'),
#                                           np.datetime64('2005-03-01'),
#                                           np.datetime64('2005-04-01'),
#                                           np.datetime64('2005-05-01'),
#                                           np.datetime64('2005-06-01'),
#                                           np.datetime64('2005-07-01'),
#                                           np.datetime64('2005-07-01'),
#                                           np.datetime64('2005-07-01'),],axis=None),
#                                 'DISDATE':
#                                     np.concatenate(
#                                           [np.datetime64(None),
#                                           np.datetime64(None),
#                                           np.datetime64('2005-03-01'),
#                                           np.datetime64(None),
#                                           np.datetime64(None),
#                                           np.datetime64('2005-06-01'),
#                                           np.datetime64(None),
#                                           np.datetime64(None),
#                                           np.datetime64('2005-07-01'),],axis=None),
#                                })
    
    
#     def test_remove_subjects_disdate_before_admidate(self):
#         df_out = clean_hes.remove_subjects_disdate_before_admidate(self.df.copy())
#         df_exp = self.df.copy()
#         pd.testing.assert_frame_equal(df_exp, df_out)
    
#     def test_remove_subjects_disdate_before_admidate_invalid(self):
        
#         self.df['DISDATE'] = np.concatenate(
#                                           [np.datetime64(None),
#                                           np.datetime64(None),
#                                           np.datetime64('2005-03-01'),
#                                           np.datetime64(None),
#                                           np.datetime64(None),
#                                           np.datetime64('2005-05-01'),
#                                           np.datetime64(None),
#                                           np.datetime64(None),
#                                           np.datetime64('2005-06-01'),],axis=None)
#         df_out = clean_hes.remove_subjects_disdate_before_admidate(self.df.copy())
#         df_exp = self.df.loc[self.df['ENCRYPTED_HESID']==0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_exp, df_out)




# class Test_full_runthrough(unittest.TestCase):
    
    
#     def setUp(self):

#         # # this is necessary to refresh all param options
#         # import importlib; from pipeline_hes import params; importlib.reload(params)
        

#         times = np.array([np.datetime64('2005-03-01'),
#                           np.datetime64('2005-01-01'),
#                           np.datetime64('2005-04-01'),
#                           np.datetime64('2005-02-01'),
#                           np.datetime64('2010-06-01'),
#                           np.datetime64('2010-07-01'),
#                           np.datetime64('2010-08-01'),
#                           np.datetime64('2005-06-01'),
#                           np.datetime64('2005-07-01'),
#                           np.datetime64('2005-08-01'),
#                           np.datetime64('2005-09-01'),
#                           np.datetime64('2005-10-01'),
#                           np.datetime64('2005-11-01')])
#         matched = np.array([np.datetime64('2005-01-01'),
#                             np.datetime64('2005-01-01'),
#                             np.datetime64('2005-01-01'),
#                             np.datetime64('2005-01-01'),
#                             np.datetime64('2010-07-01'),
#                             np.datetime64('2010-07-01'),
#                             np.datetime64('2010-07-01'),
#                             np.datetime64('2005-07-01'),
#                             np.datetime64('2005-07-01'),
#                             np.datetime64('2005-07-01'),
#                             np.datetime64('2005-10-01'),
#                             np.datetime64('2005-10-01'),
#                             np.datetime64('2005-10-01')])

#         # CONTROL WITH AMI (Alive)
#         # Control WITH AMI (Dead - but will be treated as an alive subject)
#         # Control without AMI
#         # AMI patient
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,0,
#                                                     1,1,1,
#                                                     2,2,2,
#                                                     3,3,3],
#                                 'MYADMIDATE':times,
#                                 'DISDATE':times,
#                                 'MYEPISTART':times,
#                                 'MYEPIEND':times,
#                                 'AMI':[True,False,True,False,
#                                         False,False,True,
#                                         False,False,False,
#                                         False,True,False],
#                                 'IS_CONTROL':[True,True,True,True,
#                                               True,True,True,
#                                               True,True,True,
#                                               False,False,False],
#                                 'MATCHED_DATE':matched,
#                                 'amiID':[10,10,10,10,
#                                           13,13,13,
#                                           11,11,11,
#                                           12,12,12],
#                                 'Mortality':[0,0,0,0,
#                                               1,1,1,
#                                               1,1,1,
#                                               0,0,0],
#                                 'SEX':[0,0,0,0,
#                                         0,0,0,
#                                         1,1,1,
#                                         1,1,1],
#                                 'MYDOB':np.append(
#                                     np.repeat(np.datetime64('2000-01-01'),4),
#                                     [np.repeat(np.datetime64('1940-01-01'),3),
#                                       np.repeat(np.datetime64('1990-01-01'),3),
#                                       np.repeat(np.datetime64('1970-01-01'),3),]),
#                                 'IMD04':[0,0,0,0,
#                                           4,4,4,
#                                           3,3,3,
#                                           1,1,1],
#                                 'PROCODE':[2,2,2,2,
#                                             1,1,1,
#                                             0,0,0,
#                                             2,2,2],
#                                 'SURVIVALTIME':[np.nan,np.nan,np.nan,np.nan,
#                                                 50,30,20,
#                                                 90,60,30,
#                                                 np.nan,np.nan,np.nan,],
#                                 'PROVSPNOPS':[0,1,2,3,
#                                               2,3,4,
#                                               4,5,6,
#                                               6,7,8],
#                                 'EPIORDER':[1,1,1,1,
#                                             1,1,1,
#                                             1,1,1,
#                                             1,1,1,],
#                                 'EPISTAT':[3,3,3,3,
#                                             3,3,3,
#                                             3,3,3,
#                                             3,3,3,],
#                                 'ADMIMETH':['0','0','0','0',
#                                             '0','0','0',
#                                             '0','0','0',
#                                             '0','0','0',]
#                                 })
                
#         diags_pri = ['I21','I22','I23','I211',
#                       'A00','B00','i21A',
#                       'I24','O11','a000',
#                       'A00','B00','N99']
#         diag_sec = ['B00','C00','X00','I211',
#                     'A00','B00','K24',
#                     'J42','O11','a000',
#                     'A00','D49','N99']
#         self.df ['DIAG_01'] = diags_pri
#         self.df  = self.df.astype({'DIAG_01':'category'})
#         for d in params.SEC_DIAG_COLS:
#             self.df[d] = diag_sec
#             self.df = self.df.astype({d:'category'})
    
    
#     # def tearDown(self):
#     #     from pipeline_hes.params import params
    
#     # def _get_params_list(self):
#     #     from pipeline_hes import params
#     #     return [
#     #         params.Params(),
#     #         params.Params(CHAPTER_HEADINGS_USE_GRANULAR=True),
#     #         params.Params(USE_SEC_DIAGS_IN_TRACES=True),
#     #         params.Params(RARE_EVENT_PRC_THRESH=0.001),
#     #         params.Params(LIMIT_TO_TIME_IGNORE_THESE='<6m'),
#     #         params.Params(LIMIT_TO_TIME_IGNORE_THESE='>6m')] 
    

    
#     def test_full_runthrough(self):
#         self._sub_test_full_runthrough()


#     # @patch('pipeline_hes.params.params.CONVERT_AMI_CODE_IN_TRACES', True)
#     # def test_full_runthrough2(self):
#     #     self._sub_test_full_runthrough()
            

#     # def test_full_runthrough3(self):
#     #     self._sub_test_full_runthrough()

#     @patch('pipeline_hes.params.params.SKIP_SAVING_COUNTS', True)
#     @patch('pipeline_hes.params.params.R', 'full_runthrough')
#     def _sub_test_full_runthrough(self):

#         self.df.to_parquet('{}LOADED.txt_{}_.gzip'.format(params.S_DIR,params.R),
#                         compression='gzip')

#         clean_hes.main()
#         df_out = pd.read_parquet('{}CLEAN.txt_{}_.gzip'.format(params.SAVE_FOLDER,params.R))

# #        print(df_out)

#         # ##############
#         # check the CHANGED columns,
#         # ##############
#         np.testing.assert_array_equal(df_out['DIAG_01'],
#                                       ['I21','I22','I23','I21',
#                                         'A00','B00','I21',
#                                         'I24','O11','A00',
#                                         'A00','B00','N99'])
#         np.testing.assert_array_equal(df_out['DIAG_02'],
#                                       ['B00','C00','X00','I21',
#                                         'A00','B00','K24',
#                                         'J42','O11','A00',
#                                         'A00','D49','N99'])
#         np.testing.assert_array_equal(df_out['PROCODE'],
#                                       [2,2,2,2,
#                                         1,1,1,
#                                         0,0,0,
#                                         2,2,2])
#         np.testing.assert_array_equal(df_out['PROVSPNOPS'],
#                                       [0,1,2,3,
#                                         2,3,4,
#                                         4,5,6,
#                                         6,7,8])
        
        
        
        
#         # control with AMI will be set to NOT DEAD
#         np.testing.assert_array_equal(df_out['Mortality'],
#                                       [0,0,0,0,
#                                         0,0,0,
#                                         1,1,1,
#                                         0,0,0])
#         # ###########
#         # check NEW columns
#         # ###########
#         np.testing.assert_array_equal(df_out['MYEPISTART_FIRSTAMI'],
#                                       np.append(np.repeat(np.datetime64('2005-03-01'),4),
#                                                 [np.repeat(np.datetime64('2010-08-01'),3),
#                                                   np.repeat(np.datetime64(None),3),
#                                                   np.repeat(np.datetime64('2005-10-01'),3)]))
        
#         np.testing.assert_array_equal(df_out['MATCHED_DATE'],
#                                       np.append(np.repeat(np.datetime64('2005-01-01'),4),
#                                                 [np.repeat(np.datetime64('2010-07-01'),3),
#                                                   np.repeat(np.datetime64('2005-07-01'),3),
#                                                   np.repeat(np.datetime64('2005-10-01'),3)]))

#         np.testing.assert_array_equal(df_out['INIT_AGE'],
#                                       [ 5.  ,  5.  ,  5.  ,  5.  , 70.5 , 70.5 , 70.5 , 15.5 , 15.5 ,
#                                         15.5 , 35.75, 35.75, 35.75])


#         mean_expected_death = pd.Series([np.datetime64('2005-08-01')+pd.to_timedelta(31/2,'D')+np.timedelta64(30,'D')]).round('S')

#         np.testing.assert_array_equal(df_out['DEATHDATE'],
#                                       np.array(
#                                           np.concatenate([np.repeat(np.datetime64(None),4),
#                                                           np.repeat(np.datetime64(None),3),
#                                                           np.repeat(mean_expected_death,3),
#                                                           np.repeat(np.datetime64(None),3)],axis=None),
#                                               dtype='datetime64[ns]'))



# class Test_remove_subjects_invalid(unittest.TestCase):
    
#     def setUp(self):
#         times = np.arange('2005-01', '2005-10', dtype='datetime64[M]')
        
#         self.df = pd.DataFrame({'ENCRYPTED_HESID': [0,0,0,1,1,1,2,2,2],
#                                 'IS_CONTROL':[False,False,False,
#                                               True,True,True,
#                                               True,True,True],
#                                 'EPIORDER':[1,1,1,
#                                             1,1,1,
#                                             1,1,1], 
#                                 'MYADMIDATE':times,
#                                 'MYEPISTART':times,
#                                 'MYEPIEND':times,
#                                 'DISDATE':times,
#                                 'INRANGE_DATE':np.repeat(True,9),
#                                 'INRANGE_DIAG':np.repeat(True,9),
#                                 'INRANGE_EPIORDER':np.repeat(True,9),})
        
#         self.df['MATCHED_DATE'] = np.append(np.repeat(np.datetime64('2005-02-01'),3),
#                                               [np.repeat(np.datetime64('2005-05-01'),3),
#                                               np.repeat(np.datetime64('2005-08-01'),3)])
#         self.df['MYEPISTART_FIRSTAMI'] = np.append(np.repeat(np.datetime64('2005-02-01'),3),
#                                                    [np.repeat(np.datetime64(None),3),
#                                                    np.repeat(np.datetime64('2005-08-01'),3)])
        
        
#         diags = ['A00','B50','C99',
#                  'A00','B50','C99',
#                  'A00','B50','C99']
#         self.df ['DIAG_01'] = diags
#         self.df  = self.df.astype({'DIAG_01':'category'})
#         for d in params.SEC_DIAG_COLS:
#             self.df [d] = diags
#             self.df = self.df.astype({d:'category'})
            
        
#         self.df_empty = self.df.drop(range(9))
#         self.df_empty.set_index(np.array([],dtype=object),inplace=True)
    
#     # No change (good)
#     # def test_remove_subjects_invalid_admimeth_noChange(self):
#     #     df_out = clean_hes.remove_subjects_invalid_admimeth(self.df.copy())
#     #     pd.testing.assert_frame_equal(self.df, df_out)

#     # def test_remove_subjects_invalid_epistat_noChange(self):
#     #     df_out = clean_hes.remove_subjects_invalid_epistat(self.df.copy())
#     #     pd.testing.assert_frame_equal(self.df, df_out)

#     def test_remove_subjects_invalid_epiorder_noChange(self):
#         df_out = clean_hes.remove_subjects_invalid_epiorder(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)

#     def test_remove_subjects_invalid_date_noChange(self):
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)


#     # #####################
#     # Bad
#     # #####################
#     # def test_remove_subjects_invalid_admimeth_bad1(self):
#     #     self.df.loc[0,'ADMIMETH'] = '99'
#     #     df_out = clean_hes.remove_subjects_invalid_admimeth(self.df.copy())
#     #     df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#     #     pd.testing.assert_frame_equal(df_expected, df_out)

#     # def test_remove_subjects_invalid_admimeth_bad2(self):
#     #     self.df.loc[0,'ADMIMETH'] = 'nan'
#     #     df_out = clean_hes.remove_subjects_invalid_admimeth(self.df.copy())
#     #     df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#     #     pd.testing.assert_frame_equal(df_expected, df_out)


#     # def test_remove_subjects_invalid_epistat_bad1(self):
#     #     self.df.loc[0,'EPISTAT'] = 'nan'
#     #     df_out = clean_hes.remove_subjects_invalid_epistat(self.df.copy())
#     #     df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#     #     pd.testing.assert_frame_equal(df_expected, df_out)


#     def test_remove_subjects_invalid_epiorder_bad1(self):
#         self.df.loc[0,'EPIORDER'] = '99'
#         df_out = clean_hes.remove_subjects_invalid_epiorder(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)

#     def test_remove_subjects_invalid_epiorder_bad2(self):
#         self.df.loc[0,'EPIORDER'] = 'nan'
#         df_out = clean_hes.remove_subjects_invalid_epiorder(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)


#     def test_remove_subjects_invalid_date_bad1_myadmidate(self):
#         self.df.loc[0,'MYADMIDATE'] = np.nan
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)


#     def test_remove_subjects_invalid_date_bad1_epistart(self):
#         self.df.loc[0,'MYEPISTART'] = np.nan
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)


#     def test_remove_subjects_invalid_date_bad1_epiend(self):
#         self.df.loc[0,'MYEPIEND'] = np.nan
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)

#     def test_remove_subjects_invalid_date_bad1_disdate(self):
#         self.df.loc[0,'DISDATE'] = np.nan
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         df_expected = self.df.copy()
#         pd.testing.assert_frame_equal(df_expected, df_out)


#     # #####################
#     # Bad but not in-range
#     # #####################
#     # def test_remove_subjects_invalid_admimeth_bad1_outside_range(self):
#     #     self.df.loc[0,'ADMIMETH'] = '99'
#     #     self.df.loc[0,'INRANGE_DATE'] = False
#     #     df_out = clean_hes.remove_subjects_invalid_admimeth(self.df.copy())
#     #     pd.testing.assert_frame_equal(self.df, df_out)

#     # def test_remove_subjects_invalid_admimeth_bad2_outside_range(self):
#     #     self.df.loc[0,'ADMIMETH'] = 'nan'
#     #     self.df.loc[0,'INRANGE_DATE'] = False
#     #     df_out = clean_hes.remove_subjects_invalid_admimeth(self.df.copy())
#     #     pd.testing.assert_frame_equal(self.df, df_out)


#     def test_remove_subjects_invalid_diagR69_bad1_outside_range(self):
#         self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('R69X')
#         self.df['DIAG_20'] = self.df['DIAG_20'].cat.add_categories('R69')

#         self.df.loc[0,'DIAG_01'] = 'R69X'
#         self.df.loc[0,'INRANGE_DIAG'] = False
#         self.df.loc[3,'DIAG_20'] = 'R69'
#         self.df.loc[3,'INRANGE_DATE'] = False
#         df_out = clean_hes.remove_subjects_invalid_diag(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']==2].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)

#     # Shouldnt have NAN in ***PRIMARY*** diag!
#     def test_remove_subjects_invalid_diag01nan(self):
#         self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('nan')
#         self.df.loc[0,'DIAG_01'] = 'nan'
#         df_out = clean_hes.remove_subjects_invalid_diag(self.df.copy())
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)
        
#     # NAN in secondary is fine (of course)
#     def test_remove_subjects_invalid_diag02nan(self):
#         self.df['DIAG_02'] = self.df['DIAG_02'].cat.add_categories('nan')
#         self.df.loc[0,'DIAG_02'] = 'nan'
#         df_out = clean_hes.remove_subjects_invalid_diag(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)

#     # def test_remove_subjects_invalid_epistat_bad1_outside_range(self):
#     #     self.df.loc[0,'EPISTAT'] = 'nan'
#     #     self.df.loc[0,'INRANGE_ADMIMETH'] = False
#     #     df_out = clean_hes.remove_subjects_invalid_epistat(self.df.copy())
#     #     pd.testing.assert_frame_equal(self.df, df_out)


#     def test_remove_subjects_invalid_epiorder_bad1_outside_range(self):
#         self.df.loc[0,'EPIORDER'] = '99'
#         self.df.loc[0,'INRANGE_DIAG'] = False
#         df_out = clean_hes.remove_subjects_invalid_epiorder(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)

#     def test_remove_subjects_invalid_epiorder_bad2_outside_range(self):
#         self.df.loc[0,'EPIORDER'] = 'nan'
#         self.df.loc[0,'INRANGE_DATE'] = False
#         df_out = clean_hes.remove_subjects_invalid_epiorder(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)


#     def test_remove_subjects_invalid_date_bad1_outside_range_myepistart(self):
#         self.df.loc[0,'MYEPISTART'] = np.nan
#         self.df.loc[0,'INRANGE_DIAG'] = False
#         self.df.loc[4,'MYEPISTART'] = np.datetime64('1801-01-01')
#         self.df.loc[4,'INRANGE_DIAG'] = False
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)

#     def test_remove_subjects_invalid_date_bad1_outside_range_myepistart(self):
#         self.df.loc[0,'MYEPISTART'] = np.nan
#         self.df.loc[0,'INRANGE_DIAG'] = False
#         self.df.loc[4,'MYEPISTART'] = np.datetime64('1801-01-01')
#         self.df.loc[4,'INRANGE_DIAG'] = False
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)

#     def test_remove_subjects_invalid_date_bad1_outside_range_disdate(self):
#         self.df.loc[0,'DISDATE'] = np.nan
#         self.df.loc[0,'INRANGE_DIAG'] = False
#         self.df.loc[4,'DISDATE'] = np.datetime64('1801-01-01')
#         self.df.loc[4,'INRANGE_DIAG'] = False
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)



#     # #####################
#     # Bad ENTIRE
#     # #####################
#     # def test_remove_subjects_invalid_admimeth_bad1_entire(self):
#     #     self.df.loc[0,'ADMIMETH'] = '99'
#     #     self.df.loc[3,'ADMIMETH'] = '99'
#     #     self.df.loc[6,'ADMIMETH'] = '99'
#     #     df_out = clean_hes.remove_subjects_invalid_admimeth(self.df.copy())
#     #     pd.testing.assert_frame_equal(self.df_empty, df_out, check_dtype=False)

#     # def test_remove_subjects_invalid_admimeth_bad2_entire(self):
#     #     self.df.loc[0,'ADMIMETH'] = 'nan'
#     #     self.df.loc[3,'ADMIMETH'] = 'nan'
#     #     self.df.loc[6,'ADMIMETH'] = np.nan
#     #     df_out = clean_hes.remove_subjects_invalid_admimeth(self.df.copy())
#     #     pd.testing.assert_frame_equal(self.df_empty, df_out, check_dtype=False)


#     def test_remove_subjects_invalid_diag_bad1(self):
#         self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('R69')
#         self.df['DIAG_02'] = self.df['DIAG_02'].cat.add_categories('R69X')
#         self.df['DIAG_20'] = self.df['DIAG_20'].cat.add_categories('R69X3')
#         self.df.loc[0,'DIAG_01'] = 'R69'
#         self.df.loc[3,'DIAG_02'] = 'R69X'
#         self.df.loc[6,'DIAG_20'] = 'R69X3'
#         df_out = clean_hes.remove_subjects_invalid_diag(self.df.copy())
#         pd.testing.assert_frame_equal(self.df_empty, df_out,
#                                       check_dtype=False, check_categorical=False)

#     def test_remove_subjects_invalid_diag_bad2(self):
#         self.df['DIAG_01'] = self.df['DIAG_01'].cat.add_categories('nan')
#         self.df['DIAG_02'] = self.df['DIAG_02'].cat.add_categories('nan')
#         #self.df['DIAG_20'] = self.df['DIAG_20'].cat.add_categories('nan')
#         self.df.loc[0,'DIAG_01'] = 'nan'
#         self.df.loc[3,'DIAG_02'] = 'nan'
#         self.df.loc[6,'DIAG_20'] = np.nan
#         df_out = clean_hes.remove_subjects_invalid_diag(self.df.copy())
#         pd.testing.assert_frame_equal(self.df.loc[self.df['ENCRYPTED_HESID']!=0].reset_index(drop=True),
#                                       df_out, check_dtype=False, check_categorical=False)


#     # def test_remove_subjects_invalid_epistat_bad1_entire(self):
#     #     self.df.loc[0,'EPISTAT'] = 'nan'
#     #     self.df.loc[3,'EPISTAT'] = 'nan'
#     #     self.df.loc[6,'EPISTAT'] = 'nan'
#     #     df_out = clean_hes.remove_subjects_invalid_epistat(self.df.copy())
#     #     pd.testing.assert_frame_equal(self.df_empty, df_out, check_dtype=False)


#     def test_remove_subjects_invalid_epiorder_bad1_entire(self):
#         self.df.loc[0,'EPIORDER'] = '99'
#         self.df.loc[3,'EPIORDER'] = '99'
#         self.df.loc[6,'EPIORDER'] = '99'
#         df_out = clean_hes.remove_subjects_invalid_epiorder(self.df.copy())
#         pd.testing.assert_frame_equal(self.df_empty, df_out, check_dtype=False)

#     def test_remove_subjects_invalid_epiorder_bad2_entire(self):
#         self.df.loc[0,'EPIORDER'] = 'nan'
#         self.df.loc[3,'EPIORDER'] = 'nan'
#         self.df.loc[6,'EPIORDER'] = np.nan
#         df_out = clean_hes.remove_subjects_invalid_epiorder(self.df.copy())
#         pd.testing.assert_frame_equal(self.df_empty, df_out, check_dtype=False)


#     def test_remove_subjects_invalid_date_bad1_entire_admidate(self):
#         self.df.loc[0,'MYADMIDATE'] = np.nan
#         self.df.loc[3,'MYADMIDATE'] = np.datetime64('1801-01-01')
#         self.df.loc[6,'MYADMIDATE'] = np.datetime64('1800-01-01')
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         pd.testing.assert_frame_equal(self.df_empty, df_out, check_dtype=False)


#     def test_remove_subjects_invalid_date_bad1_entire_epistart(self):
#         self.df.loc[0,'MYEPISTART'] = np.nan
#         self.df.loc[3,'MYEPISTART'] = np.datetime64('1801-01-01')
#         self.df.loc[6,'MYEPISTART'] = np.datetime64('1800-01-01')
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         pd.testing.assert_frame_equal(self.df_empty, df_out, check_dtype=False)


#     def test_remove_subjects_invalid_date_bad1_entire_epiend(self):
#         self.df.loc[0,'MYEPIEND'] = np.nan
#         self.df.loc[3,'MYEPIEND'] = np.datetime64('1801-01-01')
#         self.df.loc[6,'MYEPIEND'] = np.datetime64('1800-01-01')
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         pd.testing.assert_frame_equal(self.df_empty, df_out, check_dtype=False)


#     def test_remove_subjects_invalid_date_bad1_entire_disdate(self):
#         self.df.loc[0,'DISDATE'] = np.nan
#         self.df.loc[3,'DISDATE'] = np.datetime64('1801-01-01')
#         self.df.loc[6,'DISDATE'] = np.datetime64('1800-01-01')
#         df_out = clean_hes.remove_subjects_invalid_dates(self.df.copy())
#         pd.testing.assert_frame_equal(self.df.loc[self.df['ENCRYPTED_HESID']==0].\
#                                       reset_index(drop=True), df_out, check_dtype=False)


    
#     # remove rows with nan values
#     def test_remove_nan_rows_date(self):
#         df_out = clean_hes.remove_nan_rows_date(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)
        
        
#     def test_remove_nan_rows_date_change1_admidate(self):
#         self.df.loc[0,'MYADMIDATE'] = np.nan
#         df_out = clean_hes.remove_nan_rows_date(self.df.copy())
#         np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'].values,
#                                       [0,0,1,1,1,2,2,2])

#     def test_remove_nan_rows_date_change1_epistart(self):
#         self.df.loc[0,'MYEPISTART'] = np.nan
#         df_out = clean_hes.remove_nan_rows_date(self.df.copy())
#         np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'].values,
#                                       [0,0,1,1,1,2,2,2])

#     def test_remove_nan_rows_date_change1_epiend(self):
#         self.df.loc[0,'MYEPIEND'] = np.nan
#         df_out = clean_hes.remove_nan_rows_date(self.df.copy())
#         np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'].values,
#                                       [0,0,1,1,1,2,2,2])



# class Test_check_single_option_within_spells_sub(unittest.TestCase):
    
#     def setUp(self):
#         self.df = pd.DataFrame({
#             'ENCRYPTED_HESID':[0,0,0,
#                                1,1,1,
#                                2,2,2,
#                                3,3,3],
#             'PROVSPNOPS':['AXX','AXX','AXX',
#                           'AXX','AXX','AXX',
#                           'AXX','AXX','AXX',
#                           'AXX','AXX','AXX',],
#             'MYADMIDATE':np.append(np.repeat(np.datetime64('2005-01-01'),3),
#                                    [np.repeat(np.datetime64('1990-01-01'),3),
#                                     np.repeat(np.datetime64('1980-12-25'),3),
#                                     np.repeat(np.datetime64('1970-01-01'),3),])})
    
    
#     def test_check_single_option_within_spells(self):
#         df_out = clean_hes.check_single_option_within_spells(self.df.copy())
#         pd.testing.assert_frame_equal(self.df,df_out)


#     def test_check_single_option_within_subjects_change_admission_date_ok(self):
#         self.df.loc[0,'MYADMIDATE'] = np.datetime64('2005-01-02')
#         self.df.loc[0,'PROVSPNOPS'] = 'BXX'
#         self.df.loc[3,'MYADMIDATE'] = np.nan
#         self.df.loc[3,'PROVSPNOPS'] = 'BXX'
#         df_out = clean_hes.check_single_option_within_spells(self.df.copy())
#         pd.testing.assert_frame_equal(self.df, df_out)
        
        
#     def test_check_single_option_within_subjects_change_admission_date_bad(self):
#         self.df.loc[0,'MYADMIDATE'] = np.datetime64('2005-01-02')
#         self.df.loc[3,'MYADMIDATE'] = np.nan
#         df_out = clean_hes.check_single_option_within_spells(self.df.copy())
#         np.testing.assert_array_equal(df_out['ENCRYPTED_HESID'],
#                                       [2,2,2,3,3,3])


        
# class Test_remove_subjects_with_increasing_survivaltime(unittest.TestCase):

#     def setUp(self):
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
#                                                    0,0,0,
#                                                    1,1,1,
#                                                    1,1,1],
#                                 'EPIORDER':[1,2,3,
#                                             1,2,3,
#                                             1,2,3,
#                                             1,2,3],
#                                 'MYEPISTART':np.concatenate([
#                                    np.repeat(np.datetime64('2005-01-01'),3),
#                                    np.repeat(np.datetime64('2005-04-01'),3),
#                                    np.repeat(np.datetime64('2005-07-01'),3),
#                                    np.repeat(np.datetime64('2005-10-01'),3),],axis=None),
#                                 'SURVIVALTIME':[60,50,40,
#                                                 30,20,10,
#                                                 60,50,40,
#                                                 30,20,10,],
#                                 'Mortality':[1,1,1,
#                                              1,1,1,
#                                              1,1,1,
#                                              1,1,1],
#                                 'PROVSPNOPS':['A','A','A',
#                                               'B','B','B',
#                                               'A','A','A',
#                                               'B','B','B',],})        
#         self.df = self.df.astype(dtype={'SURVIVALTIME':float})

        

#     def _assert_ok(self):
#         df_tmp = clean_hes_prepare.add_mean_spell_times(self.df.copy())
#         df_out = clean_hes.remove_subjects_with_increasing_survivaltime(df_tmp.copy())
#         df_expected = df_tmp.copy()
#         pd.testing.assert_frame_equal(df_expected, df_out)
        
#     def _assert_bad(self):
#         df_tmp = clean_hes_prepare.add_mean_spell_times(self.df.copy())
#         df_out = clean_hes.remove_subjects_with_increasing_survivaltime(df_tmp.copy())
#         df_expected = df_tmp.loc[df_tmp['ENCRYPTED_HESID']==1].reset_index(drop=True)
#         pd.testing.assert_frame_equal(df_expected, df_out)

#     # ###############

#     def test_remove_subjects_with_increasing_survivaltime(self):
#         self._assert_ok()
        
#     def test_remove_subjects_with_increasing_survivaltime_ok1(self):
#         # this is still ok
#         self.df.loc[0:5,'SURVIVALTIME'] = [60,60,60,
#                                             60,60,60]
#         self._assert_ok()

#     def test_remove_subjects_with_increasing_survivaltime_ok2(self):
#         # this is still ok
#         self.df.loc[0:5,'SURVIVALTIME'] = [60,50,30,
#                                             30,20,10,]
#         self._assert_ok()


        
#     def test_remove_subjects_with_increasing_survivaltime_ok3(self):
#         # this is still ok
#         self.df.loc[0:5,'SURVIVALTIME'] = [11,11,11,
#                                             10,10,10]
#         self._assert_ok()
        
#     def test_remove_subjects_with_increasing_survivaltime_ok4(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [10,10,
#                                             10,10,
#                                             10,9.99]
#         self.df.loc[0:5,'PROVSPNOPS'] = ['A','A',
#                                           'A','B',
#                                           'B','B']
#         self.df.loc[0:5,'EPIORDER'] = [1,2,
#                                         3,1,
#                                         2,3]
#         self._assert_ok()
        
#     # ###############

#     def test_remove_subjects_with_increasing_survivaltime_bad1(self):
#         # this is NOT ok (60-50-40(WRONG_POS)-30-10)
#         self.df.loc[0:5,'SURVIVALTIME'] = [60,50,30,
#                                             30,40,10,]
#         self._assert_bad()

#     def test_remove_subjects_with_increasing_survivaltime_bad2(self):
#         # this is NOT ok 
#         self.df.loc[0:5,'SURVIVALTIME'] = [60,10,40,
#                                             11,11,11,]
#         self._assert_bad()
        
        
#     def test_remove_subjects_with_increasing_survivaltime_bad3(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [11,10,10,
#                                             10,10,11]
#         self._assert_bad()

#     def test_remove_subjects_with_increasing_survivaltime_bad4(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [11,11,
#                                             11,10,
#                                             11,10]
#         self.df.loc[0:5,'PROVSPNOPS'] = ['A','A',
#                                           'B','B',
#                                           'C','C']
#         self._assert_bad()
        

#     def test_remove_subjects_with_increasing_survivaltime_bad5(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [11,11,
#                                             10,10,
#                                             11,11.1]
#         self.df.loc[0:5,'PROVSPNOPS'] = ['A','A',
#                                           'B','B',
#                                           'C','B']
#         self.df.loc[0:5,'EPIORDER'] = [1,2,
#                                         1,2,
#                                         1,3]
#         self._assert_bad()
    

#     def test_remove_subjects_with_increasing_survivaltime_bad6(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [11,10,
#                                             10.5,10.5,
#                                             11,11]
#         self.df.loc[0:5,'PROVSPNOPS'] = ['A','A',
#                                           'B','B',
#                                           'C','C']
#         self.df.loc[0:5,'EPIORDER'] = [1,2,
#                                         1,2,
#                                         1,2]
#         self._assert_bad()
        
        
#     def test_remove_subjects_with_increasing_survivaltime_bad7(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [11,10,
#                                             9,10,
#                                             10,10]
#         self.df.loc[0:5,'PROVSPNOPS'] = ['A','A',
#                                           'A','B',
#                                           'B','B']
#         self.df.loc[0:5,'EPIORDER'] = [1,2,
#                                         3,1,
#                                         2,3]
#         self._assert_bad()
        
        
        
#     def test_remove_subjects_with_increasing_survivaltime_bad8(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [11,10,
#                                             9,12,
#                                             10,8]
#         self.df.loc[0:5,'PROVSPNOPS'] = ['A','A',
#                                           'A','B',
#                                           'B','B']
#         self.df.loc[0:5,'EPIORDER'] = [1,2,
#                                         3,1,
#                                         2,3]
#         self._assert_bad()
        
#     def test_remove_subjects_with_increasing_survivaltime_bad9(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [10,10,
#                                             10,10.01,
#                                             10,9.99]
#         self.df.loc[0:5,'PROVSPNOPS'] = ['A','A',
#                                           'A','B',
#                                           'B','B']
#         self.df.loc[0:5,'EPIORDER'] = [1,2,
#                                         3,1,
#                                         2,3]
#         self._assert_bad()

#     def test_remove_subjects_with_increasing_survivaltime_bad10(self):
#         # this is NOT ok (epiorder forces order)
#         self.df.loc[0:5,'SURVIVALTIME'] = [60,10,40,
#                                             10,10,10,]
#         self._assert_bad()

#     def test_remove_subjects_with_increasing_survivaltime_bad11(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = [11,11,
#                                             10,10,
#                                             11,11]
#         self.df.loc[0:5,'PROVSPNOPS'] = ['A','A',
#                                           'B','B',
#                                           'C','B']
#         self.df.loc[0:5,'EPIORDER'] = [1,2,
#                                         1,2,
#                                         1,3]
#         self._assert_bad()

#     # #########
#     # nans
#     # #########
#     def test_remove_subjects_with_increasing_survivaltime_nan_one_sub(self):
#         self.df.loc[0:5,'SURVIVALTIME'] = np.nan
#         self._assert_ok()
        
#     def test_remove_subjects_with_increasing_survivaltime_nan_all_subs(self):
#         self.df['SURVIVALTIME'] = np.nan
#         self._assert_ok()
        
#     def test_remove_subjects_with_increasing_survivaltime_mortality0(self):
#         self.df['Mortality'] = 0
#         self._assert_ok()
        
#     def test_remove_subjects_with_increasing_survivaltime_mortality0_nan(self):
#         self.df['Mortality'] = 0
#         self.df['SURVIVALTIME'] = np.nan
#         self._assert_ok()
        

        
# class Test_remove_using_survivaltime_mortality(unittest.TestCase):
    
#     def setUp(self):
#         self.df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,
#                                                    1,1,1,
#                                                    2,2,2,
#                                                    3,3,3],
#                            'Mortality':[1,1,1,
#                                         1,1,1,
#                                         0,0,0,
#                                         1,1,1],
#                            'SURVIVALTIME':[3,4,5,
#                                            3,4,np.nan,
#                                            np.nan,np.nan,np.nan,
#                                            np.nan,6,np.nan],
#                            'IS_CONTROL':[0,0,0,
#                                          0,0,0,
#                                          0,0,0,
#                                          1,1,1],
#                            'MYEPISTART_FIRSTAMI':np.concatenate([
#                                    np.repeat(np.datetime64('2005-01-01'),3),
#                                    np.repeat(np.datetime64('2005-04-01'),3),
#                                    np.repeat(np.datetime64('2005-07-01'),3),
#                                    np.repeat(np.datetime64('2005-10-01'),3),],axis=None)})
        
#         self.df = self.df.astype(dtype={'ENCRYPTED_HESID':object,
#                                     'Mortality':np.uint8,
#                                     'SURVIVALTIME':float})
    
#     def test_remove_mixed_survival_mortality_alive(self):
        
#         df_out = clean_hes.remove_mixed_survival_mortality(self.df.copy())
#         df_out = df_out.sort_values('ENCRYPTED_HESID').reset_index(drop=True)
        
#         # controls with AMI are always kept
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID']!=1].reset_index(drop=True)
#         df_expected = df_expected.astype({'ENCRYPTED_HESID':object,
#                                           'Mortality':np.uint8,
#                                           'SURVIVALTIME':float})
#         df_expected.index = list(df_expected.index)
        
#         pd.testing.assert_frame_equal(df_expected, df_out)
    
#     def test_remove_mixed_survival_mortality_dead(self):
        
#         self.df['Mortality'] = np.uint8(1)        
#         df_out = clean_hes.remove_mixed_survival_mortality(self.df.copy())
        
#         # controls with AMI are always kept
#         df_expected = self.df.loc[self.df['ENCRYPTED_HESID'].map(lambda x: x in [0,3])].\
#             reset_index(drop=True)
#         df_expected = df_expected.astype({'ENCRYPTED_HESID':object,
#                                           'Mortality':np.uint8,
#                                           'SURVIVALTIME':float})
        
#         pd.testing.assert_frame_equal(df_expected, df_out)



# class Test_remove_subjects_invalid_dob(unittest.TestCase):
    
#     def test_remove_subjects_invalid_dob(self):
#         df = pd.DataFrame({'ENCRYPTED_HESID':[0,0,0,1,1,1,2,2,2],
#                            'MYDOB':
#                                np.concatenate([
#                                        np.repeat(np.datetime64('2005-01-01'),3),
#                                        np.repeat(np.datetime64('2005-04-01'),3),
#                                        np.repeat(np.datetime64('2005-07-01'),3),],axis=None),
#                                })                             
#         df.loc[0:2, 'MYDOB'] = np.datetime64('1800-01-01')
#         df.loc[3:5, 'MYDOB'] = np.nan

#         df_expected = pd.DataFrame({'ENCRYPTED_HESID':[2,2,2],
#                                     'MYDOB': np.repeat(np.datetime64('2005-07-01'),3)
#                                     },index=[6,7,8])
        
#         df_out = clean_hes.remove_subjects_invalid_dob(df.copy())
#         pd.testing.assert_frame_equal(df_expected, df_out)