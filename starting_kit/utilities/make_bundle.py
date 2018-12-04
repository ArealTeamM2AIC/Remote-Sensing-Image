#########################################################################################

#				MAKE CODALAB COMPETITION BUNDLE V1

#########################################################################################

# Isabelle Guyon -- November 16, 2018

# Script to make a competition bundle from a starting kit.
# The following files should be provided in a directory "starting_kit"
# logo.jpg								# Your competition logo
# README.ipynb							# The notebook explaining how to make submissions
# README.md								# A Readme file with basic explanations
# ingestion_program/					# Code for ingestion of participant submissions				
# scoring_program/						# Code scoring the results of participant submissions	
# sample_code_submission/				# Simple example of participant submission
# sample_data/							# Small sample dataset (subset of the real training set)
# challenge_data/						# Challenge data in AutoML split format
# html_pages/							# HTML documentation pages

# The code tests the "sample_code_submission" with the ingestion and scoring program
# to verify that all works well and then compiles a challenge bundle.

# Usage: 
#		python make_bundle.py  
#		python make_bundle.py starting_kit_dir big_data_dir

import os, shutil
import datetime
from glob import glob as ls
from sys import argv, version_info
import zipfile

# Check we are running the correct version of python
if version_info < (3, 6):
	print("Wrong Python version {}, use Python 3.6".format(version_info))
	exit(0)

class Bundle:
	def __init__(self, starting_kit_dir, big_data_dir):
		''' Defines a bundle structure.'''
		# Starting kit template
		self.starting_kit_files = ['README.md', 'README.ipynb', 'scoring_program', 'ingestion_program', 'sample_code_submission', 'sample_data']
		self.starting_kit_dir = starting_kit_dir
		# Data and code
		self.big_data = big_data_dir
		self.sample_data = os.path.join(starting_kit_dir, 'sample_data')
		self.scoring_program = os.path.join(starting_kit_dir, 'scoring_program')
		self.ingestion_program = os.path.join(starting_kit_dir, 'ingestion_program')
    	# Sample submissions and outputs
		self.sample_code_submission = os.path.join(starting_kit_dir, 'sample_code_submission')
		self.sample_result_submission = os.path.join(starting_kit_dir, 'sample_result_submission') 
		self.scoring_output = os.path.join(starting_kit_dir, 'scoring_output') 
    	# Other files
		self.yamlfile = os.path.join(starting_kit_dir, 'utilities', 'competition.yaml')
		self.image = os.path.join(starting_kit_dir,'logo.jpg')  
		self.html_pages = os.path.join(starting_kit_dir, 'html_pages') 
		self.readme = os.path.join(starting_kit_dir, 'README')
		# Select the compression mode ZIP_DEFLATED for compression
		# or zipfile.ZIP_STORED to just store the file
		self.compression = zipfile.ZIP_DEFLATED
						    	
	def check(self):
		''' Checks that the starting kit structure is correct and that
		the ingestion program and scoring program work.'''
		execution_success = 1
		# Verify the structure of the starting kit
		actual_starting_kit_files = set([os.path.basename(x) for x in ls(os.path.join(self.starting_kit_dir, '*'))])
		desired_starting_kit_file =  set(self.starting_kit_files)
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("%% CHECKS %% 1/3 %% CHECKS %% 1/3 %% CHECKS %%")
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("    Checking starting kit structure:")
		print('	{}'.format(self.starting_kit_files))
		if actual_starting_kit_files & desired_starting_kit_file != desired_starting_kit_file:
			print("[-] Failed, some files are missing, got only these files:")
			print('	{}'.format(actual_starting_kit_files))
			return 0
		else:
			print("[+] Passed")
		# Add "sample_result_submission" to the list of things to deliver with the starting kit
		self.starting_kit_files.append('sample_result_submission')
		# Add and HTML version of the jupyter notebook (unfortunately this messes up the website, we don't do it for now)
		#print("    Creating HTML version of notebook:")
		#command_notebook = 'jupyter nbconvert --to html {} --stdout >> {}'.format(os.path.join(self.starting_kit_dir, 'README.ipynb'), os.path.join(self.html_pages, 'README.html'))
		#os.system(command_notebook)
		#print("[+] Done")
		# Create directories if they do not already exits
		if not os.path.isdir(self.sample_result_submission): os.mkdir(self.sample_result_submission)
		if not os.path.isdir(self.scoring_output): os.mkdir(self.scoring_output)
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("%% CHECKS %% 2/3 %% CHECKS %% 2/3 %% CHECKS %%")
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		# Run the ingestion program with sample data or big data
		path_ingestion = os.path.join(self.ingestion_program, 'ingestion.py')
		if big_data_dir:
			data_dir = self.big_data
		else:
			data_dir = self.sample_data
		command_ingestion = 'python {} {} {} {} {}'.format(path_ingestion, data_dir, self.sample_result_submission, self.ingestion_program, self.sample_code_submission)
		os.system(command_ingestion)
		# Check that predictions were made:
		results = ls(os.path.join(starting_kit_dir, '*/*.predict'))
		if len(results)!=3: 
			print("[-] Failed, some prediction files are missing, got only:")
			print('	{}'.format(results))
			return 0
		else:
			print("[+] Passed")
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("%% CHECKS %% 3/3 %% CHECKS %% 3/3 %% CHECKS %%")
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		# Run the scoring program
		path_scoring = os.path.join(self.scoring_program, 'score.py')
		command_scoring = 'python {} {} {} {}'.format(path_scoring, data_dir, self.sample_result_submission, self.scoring_output)
		os.system(command_scoring)
		# Check that scores were computed:
		scores = ls(os.path.join(starting_kit_dir, '*/scores.*'))
		if len(scores)!=2: 
			print("[-] Failed, some score files are missing, got only:")
			print('	{}'.format(scores))
			return 0
		else:
			print("[+] Passed")
		return execution_success
	
	def zip(self, destination):
		''' Creates a zip file with everything needed to create a competition.'''
		execution_success = 1
		if not os.path.isfile(destination): os.mkdir(destination)
		print("Creating bundle: {}.zip".format(destination))
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("%% ZIP %% 1/3 %% ZIP %% 1/3 %% ZIP %%")
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		# Zip relevant files and put them in destination
		execution_success *= self.compress_code(self.scoring_program, destination)
		execution_success *= self.compress_code(self.ingestion_program, destination)
		if big_data_dir:
			data_dir = self.big_data
		else:
			data_dir = self.sample_data
		execution_success *= self.compress_data(data_dir,  destination)
		# Move other relevant file to destination
		execution_success *= self.move_other_files(destination)
		# Zip the sample submissions and add them to the starting kit directory
		execution_success *= self.compress_sample_submission(self.sample_code_submission, 'sample_code_submission')
		execution_success *= self.compress_sample_submission(self.sample_code_submission, 'sample_trained_submission')
		execution_success *= self.compress_sample_submission(self.sample_result_submission, 'sample_result_submission')
		# Add the starting kit
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("%% ZIP %% 2/3 %% ZIP %% 2/3 %% ZIP %%")
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		execution_success *= self.compress_starting_kit(destination)
		# Zip the overall submission
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		print("%% ZIP %% 3/3 %% ZIP %% 3/3 %% ZIP %%")
		print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
		execution_success *= self.compress_competition_bundle(destination)
		return execution_success
    	   	
	def compress_code(self, dir_name, destination):
		''' Compress all '.py' files in dir_name. Add metadata if it exists. '''
		execution_success = 1
		file_names = ls(os.path.join(dir_name, '*.py'))
		metadata = os.path.join(dir_name, 'metadata')
		if os.path.exists(metadata): 
			file_names = file_names + [ metadata ]
		metric = os.path.join(dir_name, 'metric.txt')
		if os.path.exists(metric): 
			file_names = file_names + [ metric ]
		print('    Compressing code files:')
		print('	{}'.format(file_names))
		# create the zip file 
		[dirnm, filenm] = os.path.split(dir_name)
		zf = zipfile.ZipFile(os.path.join(destination, filenm + '.zip'), mode='w')
		try:
			for file_name in file_names:
				[dirnm, filenm] = os.path.split(file_name)
				# Add file to the zip file
				# first parameter file to zip, second filename in zip
				zf.write(file_name, filenm, compress_type=self.compression)
			print('[+] Success')
		except:
			print('[-] An error occurred while zipping code files: ' + dir_name)
			execution_success = 0
		finally:
			# Close the file
			zf.close() 	
		return execution_success
			
	def compress_data(self, dir_name, destination):
		''' Compress data files in AutoML split format. '''
		execution_success = 1
		file_names = ls(os.path.join(dir_name, '*.*'))
		print('    Compressing data files:')
		print('	{}'.format(file_names))
		# create zip files for input_data and reference data
		z_input = zipfile.ZipFile(os.path.join(destination, 'input_data.zip'), mode='w')
		z_ref1 = zipfile.ZipFile(os.path.join(destination, 'reference_data_1.zip'), mode='w')
		z_ref2 = zipfile.ZipFile(os.path.join(destination, 'reference_data_2.zip'), mode='w')
		try:
			for file_name in file_names:
				[dirnm, filenm] = os.path.split(file_name)
				# Add file to the zip file
				# first parameter file to zip, second filename in zip
				if filenm.find('valid.solution')==-1 and filenm.find('test.solution')==-1 and filenm.find('private.info')==-1:
					#print('Add {} to input'.format(filenm))
					z_input.write(file_name, filenm, compress_type=self.compression)
				if filenm.find('public.info')>=0:
					#print('Add {} to refs'.format(filenm))
					z_ref1.write(file_name, filenm, compress_type=self.compression)
					z_ref2.write(file_name, filenm, compress_type=self.compression) 
				if filenm.find('valid.solution')>=0:
					#print('Add {} to ref1'.format(filenm))
					z_ref1.write(file_name, filenm, compress_type=self.compression)
				if filenm.find('test.solution')>=0:
					#print('Add {} to ref2'.format(filenm))
					z_ref2.write(file_name, filenm, compress_type=self.compression) 
			self.starting_kit_files += ['sample_code_submission.zip', 'sample_result_submission.zip', 'sample_trained_submission.zip'] 
			print('[+] Success')          	
		except:
			print('[-] An error occurred while zipping data files: ' + dir_name)
			execution_success = 0
		finally:
			# Close the files
			z_input.close()
			z_ref1.close()
			z_ref2.close()
		return execution_success
		
	def move_other_files(self, destination):
		''' Move other relevant files. '''		
		execution_success = 1
		print('    Moving other files:')
		try:
			print("	YAML configuration file: {}".format(self.yamlfile))
			shutil.copy2(self.yamlfile, destination)
			print("	Image/logo file: {}".format(self.image))
			shutil.copy2(self.image, destination)
			html_pages = ls(os.path.join(self.html_pages, '*.html'))
			print("	HTML pages: {}".format(html_pages))
			for file in html_pages:
				shutil.copy2(file, destination)
			print('[+] Success')
		except:
			print('[-] An error occurred while copying other files to: ' + destination)
			execution_success = 0
		return execution_success
		
	def compress_sample_submission(self, dir_name, destination):
		''' Create 3 samples submisssions ready to go: one with results, 
		one with unstrained model, and one with trained model. '''
		execution_success = 1
		zf = zipfile.ZipFile(os.path.join(self.starting_kit_dir, destination + '.zip'), mode='w')
		if destination.find('result')>=0:
			# This is a prediction result directory
			file_names = ls(os.path.join(dir_name, '*.predict'))
		else:
			# This is a code directory
			file_names = ls(os.path.join(dir_name, '*.py'))
			metadata = os.path.join(dir_name, 'metadata')
			if os.path.exists(metadata): 
				file_names = file_names + [ metadata ]
			# Add the pickle?
			if destination.find('trained')==-1:
				pickle = ls(os.path.join(dir_name, '*.piclke'))
				file_names = file_names + pickle
		print('    Compressing submission files:')
		print('	{}'.format(file_names))
		# create the zip file 
		try:
			for file_name in file_names:
				[dirnm, filenm] = os.path.split(file_name)
				# Add file to the zip file
				# first parameter file to zip, second filename in zip
				zf.write(file_name, filenm, compress_type=self.compression)
			print('[+] Success')
		except:
			print('[-] An error occurred while zipping code files: ' + dir_name)
			execution_success = 0
		finally:
			# Close the file
			zf.close() 	
		return execution_success
		
	def compress_starting_kit(self, destination):
		''' Compress relevant directories and files from the starting kit. '''
		execution_success = 1
		print('    Compressing starting kit files:')
		print('	{}'.format(self.starting_kit_files))
		zf = zipfile.ZipFile(os.path.join(destination, 'starting_kit.zip'), mode='w')
		try:
			for filenm_ in self.starting_kit_files:
				# Add file to the zip file
				# first parameter file to zip, second filename in zip
				dirname = os.path.join(self.starting_kit_dir, filenm_)
				#print('	+ Adding {}'.format(dirname))
				zf.write(dirname, filenm_, compress_type=self.compression)
				if os.path.isdir(dirname):
					#print('	+ Adding {} contents:'.format(dirname))
					file_names = ls(os.path.join(dirname, '*'))
					#print(file_names)
					for file_name in file_names:
						if(file_name.find('__pycache__')==-1 and file_name.find('.pyc')==-1):
							#print(file_name)
							[dirnm, filenm] = os.path.split(file_name)
							zf.write(file_name, os.path.join(filenm_,filenm), compress_type=self.compression)
			print('[+] Success')
		except:
			print('[-] An error occurred while zipping starting kit files: ' + self.starting_kit_dir)
			execution_success = 0
		finally:
			# Close the file
			zf.close() 	
		return execution_success
		
	def compress_competition_bundle(self, destination):
		''' Compress the overall competition bundle. '''
		execution_success = 1
		print('    Compressing competition bundle: {}'.format(destination))
		zf = zipfile.ZipFile(destination + '.zip', mode='w')
		try:
			for dirname in ls(os.path.join(destination,'*')):
				[dirnm, filenm_] = os.path.split(dirname)
				# Add file to the zip file
				# first parameter file to zip, second filename in zip
				print('	+ Adding {}'.format(filenm_))
				zf.write(dirname, filenm_, compress_type=self.compression)
				if os.path.isdir(filenm_):
					print('	+ Adding {} contents:'.format(dirname))
					file_names = ls(os.path.join(dirname, '*'))
					print(file_names)
					for file_name in file_names:
						print(file_name)
						[dirnm, filenm] = os.path.split(file_name)
						zf.write(file_name, os.path.join(filenm_,filenm), compress_type=self.compression)
			print('[+] Success')
		except:
			print('[-] An error occurred while zipping the competition bundle: ' + destination)
			execution_success = 0
		finally:
			# Close the file
			zf.close() 	
		return execution_success
		
	def get_data_name(self):
		''' Get the name of the dataset.'''
		file_names = ls(os.path.join(self.sample_data, '*.*'))
		[dirnm, filenm] = os.path.split(file_names[0])
		lnm = filenm.split('_')
		return lnm[0]

#################################################
# 					MAIN PROGRAM
#################################################	
		
if __name__== "__main__":

	# Purge a whole bunch of files before we start (choose which ones):
	##############################################
	clean_old_bundles = 1			# Remove old bundles 
	clean_pyc = 1					# Remove old pyc
	clean_pycache = 1				# Clear pycache
	clean_readme = 1				# Remove old README.html
	clean_pickle = 1				# delete model pickle
	clean_results = 1				# delete prediction results
	clean_scores = 1				# delete old scores
	clean_submissions = 1			# delete old sample submissions
	
	def purge_files(list_name):
		'''Remove files in list called list_name (list_name is a string) '''
		file_list = eval(list_name)
		if len(file_list)>0: 
			print("[x] Removing {}: {}".format(list_name, file_list))
			for file in file_list:
				if os.path.isdir(file): 
					shutil.rmtree(file)
				else:
					os.remove(file)	
	
	# Input directories:
    ####################
	if len(argv)==1:
		starting_kit_dir = '../'
		big_data_dir = ''
	else:
		starting_kit_dir = os.path.abspath(argv[1])
		big_data_dir = os.path.abspath(argv[2])
		
	# Bundle object creation and checks: 
	####################################
	my_bundle = Bundle(starting_kit_dir, big_data_dir)
	
	data_name = my_bundle.get_data_name()
	the_date = datetime.datetime.now().strftime('%y-%m-%d-%H-%M')
	destination = data_name + '_bundle_' + the_date
		
	print("\n\n##############################################################")
	print("##############################################################")
	print("#\n#       Processing bundle : {}       #\n#".format(destination))
	print("##############################################################")
	print("##############################################################")
	print("\nUsing starting_kit_dir: {}".format(starting_kit_dir))
	if big_data_dir: print("Using big_data_dir: {}\n".format(big_data_dir))
	
	# Proceed to purge:	
	if clean_old_bundles:
		old_bundles = ls('*_bundle_*')
		purge_files('old_bundles')
	if clean_pyc:
		old_pyc = ls(os.path.join(starting_kit_dir, '*/*.pyc'))
		purge_files('old_pyc')
	if clean_pycache:
		old_pycache = ls(os.path.join(starting_kit_dir, '*/__pycache__'))
		purge_files('old_pycache')
	if clean_readme:
		old_readme = ls(os.path.join(starting_kit_dir, '*/README.html'))
		purge_files('old_readme')
	if clean_pickle:
		old_pickle = ls(os.path.join(starting_kit_dir, '*/*.pickle'))
		purge_files('old_pickle')
	if clean_results:
		old_results = ls(os.path.join(starting_kit_dir, '*/*.predict'))
		purge_files('old_results')
	if clean_results:
		old_scores = ls(os.path.join(starting_kit_dir, '*/scores.*'))
		purge_files('old_scores')
	if clean_submissions:
		old_submissions = ls(os.path.join(starting_kit_dir, '*_submission.zip'))
		purge_files('old_submissions')	
						
	# Perform a sanity check:
	if not my_bundle.check():
		print('[-] Something went wrong, sorry, try again!!')
		exit(0)
	
    # Zip the bundle:
	execution_success = my_bundle.zip(destination)
	
	if execution_success:
		print('[+] Congratulations, you are done!! Your bundle is called: {}'.format(destination))
	else:
		print('[-] Something went wrong, sorry, try again!!')

