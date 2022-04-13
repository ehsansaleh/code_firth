from os.path import dirname

####################################################
################# global configs ###################
####################################################

naming_scheme = 'firth'
main_acc = 'test_acc'
firth_reg_col = 'firth_coeff'
PROJPATH = f'{dirname(dirname(__file__))}'
smry_tbls_dir = f'{PROJPATH}/summary'
paper_plots_dir = f'{PROJPATH}/figures'
paper_tbls_dir = f'{PROJPATH}/tables'

####################################################
############### csv2summ configs ###################
####################################################

reg_sources = ['val']
summ_cond_vars = ['backbone_arch', 'n_shots', 'n_ways',
                  'dataset_name', 'firth_prior_type',
                  'clf_type', 'experiment']
results_csv_dir = f'{PROJPATH}/results'
# generating the path to the files to be read
# provided by a seperate .py file

csvdir_expname = [('01_firth_1layer', 'firth_1layer'),
                  ('02_l2_1layer', 'l2_1layer'),
                  ('03_imbal_1layer', 'imbal_1layer'), 
                  ('04_firth_3layer', 'firth_3layer'),
                  ('05_l2_3layer', 'l2_3layer'),
                  ('06_imbal_3layer', 'imbal_3layer'), 
                  ('07_firth_5_10way', 'firth_5_10way'),
                  ('08_firth_tieredcifar', 'firth_tieredcifar'),
                  ('09_l2_tieredcifar', 'l2_tieredcifar'), 
                  ('10_firth_densemobile', 'firth_densemobile'),
                  ('11_entropy', 'entropy')]
expname_regcol = {'firth_1layer': 'firth_coeff',
                  'l2_1layer': 'l2_coeff',
                  'imbal_1layer': 'firth_coeff',
                  'firth_3layer': 'firth_coeff',
                  'l2_3layer': 'l2_coeff',
                  'imbal_3layer': 'firth_coeff',
                  'firth_5_10way': 'firth_coeff',
                  'firth_tieredcifar': 'firth_coeff',
                  'l2_tieredcifar': 'l2_coeff',
                  'firth_densemobile': 'firth_coeff',
                  'entropy': 'ent_coeff'}

# prop_cols --> The columns which we'll be conditioned upon in the summary tables.
#               If a column is not in prop_cols, it may get treated as a "response"/"y"
#               variable, so it's best if you include all "x" columns below
#               (except rng_seed and regularization coefficient columns).
prop_cols = ['optim_type', 'n_shots', 'do_imbalanced', 'permute_labels',
             'backbone_arch', 'data_type', 'learning_rate', 'batch_size',
             'n_epochs', 'n_ways',  'n_query', 'dataset_name',
             'firth_prior_type', 'clf_type', 'experiment']
crn_cols = ['rng_seed']
dfltvals_dict = dict()

####################################################
############### summ2tables configs ################
####################################################

table_sep_cols = ['backbone_arch']
row_tree = ['n_shots']
col_tree = ['n_ways']

y_list = [(f'base_{main_acc}', f'base_{main_acc}_ci', 'Before'),
          (main_acc, f'{main_acc}_ci', 'After'),
          (f'delta_{main_acc}', f'delta_{main_acc}_ci', 'Improvement')]
scale_percent = 100
do_supplement_cols = False
