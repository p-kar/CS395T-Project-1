import os
import pdb
import subprocess as sp

OUTPUT_ROOT='/scratch/cluster/pkar/CS395T-Project-1/runs/'
SCRIPT_ROOT='/scratch/cluster/pkar/CS395T-Project-1/scripts/'

mapping_dict = {
    # Condor Scheduling Parameters
    '__EMAILID__': 'pkar@cs.utexas.edu',
    '__PROJECT__': 'INSTRUCTIONAL',
    # Algorithm hyperparameters
    '__CODE_ROOT__': '/scratch/cluster/pkar/CS395T-Project-1',
    '__MODE__': 'train',
    '__DATA_DIR__': '/scratch/cluster/pkar/CS395T-Project-1/data/yearbook',
    '__NWORKERS__': '4',
    '__BSIZE__': '32',
    '__SHUFFLE__': 'True',
    '__NCLASSES__': '120',
    '__ARCH__': 'resnet18',
    '__TARGET_TYPE__': 'classification',
    '__OPTIM__': 'adam',
    '__LR__': '1e-3',
    '__WD__': '1e-4',
    '__MOMENTUM__': '0.9',
    '__EPOCHS__': '90',
    '__MAX_NORM__': '1',
    '__LR_DECAY_STEP__': '45',
    '__LR_DECAY_GAMMA__': '0.1',
    '__START_EPOCH__': '0',
    '__LOG_ITER__': '10',
    '__RESUME__': 'False',
    '__SEED__': '123'
    }

# Figure out number of jobs to run
num_jobs = 1
for key, value in mapping_dict.iteritems():
    if type(value) == type([]):
        if num_jobs == 1:
            num_jobs = len(value)
        else:
            assert(num_jobs == len(value))

for idx in range(num_jobs):
    mapping_dict['__JOBNAME__'] = 'run_%.2d'%(idx)
    mapping_dict['__LOGNAME__'] = os.path.join(OUTPUT_ROOT, mapping_dict['__JOBNAME__'])
    mapping_dict['__LOG_DIR__'] = mapping_dict['__LOGNAME__']
    mapping_dict['__SAVE_PATH__'] = mapping_dict['__LOGNAME__']
    sp.call('mkdir %s'%(mapping_dict['__LOGNAME__']), shell=True)
    condor_script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'condor_script.sh')
    script_path = os.path.join(mapping_dict['__SAVE_PATH__'], 'run_script.sh')
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'condor_script_proto.sh'), condor_script_path), shell=True)
    sp.call('cp %s %s'%(os.path.join(SCRIPT_ROOT, 'run_proto.sh'), script_path), shell=True)
    for key, value in mapping_dict.iteritems():
        if type(value) == type([]):
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value[idx], condor_script_path), shell=True)
        else:
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, script_path), shell=True)
            sp.call('sed -i "s#%s#%s#g" %s'%(key, value, condor_script_path), shell=True)

    sp.call('condor_submit %s'%(condor_script_path), shell=True)
