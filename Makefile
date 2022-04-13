all: summary figures tables
figures: plot_dacc_vs_nshots plot_dacc_vs_bb_imbal plot_acc_vs_nshots \
         plot_dacc_vs_nways_dnmn plot_geometric_example plot_valacc_vs_lambda
prep: venv dl_lite
prep_full: prep dl_full link_tiered

.PHONY: summary venv figures tables
SHELL := /bin/bash
PROJBASE := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
MPIPRECMD := $(shell command -v mpirun >/dev/null 2>&1 && echo "mpirun -n 10")

##########################################################
###########      Summarizing CSV Results     #############
##########################################################

summary:
	source .env.sh && ${MPIPRECMD} python utils/csv2summ.py

##########################################################
####################  Figures/Tables #####################
##########################################################

plot_dacc_vs_nshots:
	source .env.sh && python plotters/plot_dacc_vs_nshots.py
plot_dacc_vs_bb_imbal:
	source .env.sh && python plotters/plot_dacc_vs_bb_imbal.py
plot_acc_vs_nshots:
	source .env.sh && python plotters/plot_acc_vs_nshots.py
plot_dacc_vs_nways_dnmn:
	source .env.sh && python plotters/plot_dacc_vs_nways_dnmn.py
plot_geometric_example:
	source .env.sh && python plotters/plot_geometric_example.py
plot_valacc_vs_lambda:
	source .env.sh && python plotters/plot_valacc_vs_lambda.py
tables:
	source .env.sh && python utils/summ2tbls.py

##########################################################
####################      VENV     #######################
##########################################################

venv:
	python -m venv venv
	source venv/bin/activate && python -m pip install --upgrade pip
	source venv/bin/activate && python -m pip install torch==1.7.1+cu101 \
		torchvision==0.8.2+cu101 \
		-f https://download.pytorch.org/whl/torch_stable.html
	source venv/bin/activate && python -m pip install -r requirements.txt

##########################################################
####################    Downloads   ######################
##########################################################

dl_lite:
	./features/download.sh

dl_full:
	./features/download.sh
	./backbones/download.sh
	./datasets/download.sh

link_tiered:
	@MSG="I need to add a shortcut (soft symlink) to the "\
	MSG=$${MSG}"imagenet dataset for tiered-imagenet."; \
	echo $$MSG; \
	MSG="Please enter the location of the your extracted "; \
	MSG=$${MSG}"imagenet dataset? "; \
	echo $$MSG; \
	read -p "[press enter to skip] " IMGNETLOC; \
	if [[ "abc"$$IMGNETLOC == "abc" ]]; then \
	  echo "Aborting"; \
	else \
	  ABSIMGNETLOC=$$(readlink -m $$IMGNETLOC); \
	  if [[ -d $$ABSIMGNETLOC"/n01440764" ]] ; then \
	    echo ln -s $$ABSIMGNETLOC datasets/tieredimagenet; \
	    ln -s $$ABSIMGNETLOC datasets/tieredimagenet; \
	  else \
	    echo $$ABSIMGNETLOC"/n01440764 is not an existing directory.\n"; \
	    echo "I will leave it to you to perform the following:"; \
	    echo "  ln -s /path/to/imagenet/ ./datasets/tieredimagenet"; \
  	  fi ;\
  	fi

##########################################################
####################      Clean     ######################
##########################################################

fix_crlf:
	find ${PROJBASE} -maxdepth 3 -type f -name "*.md5" \
	  -o -name "*.py" -o -name "*.sh" -o -name "*.json" | xargs dos2unix

clean:
	@MYPROJBASE=${PROJBASE}; \
	RESBASE=$$MYPROJBASE"/results"; \
	CSVTREE=""; \
	cd $$MYPROJBASE; \
	while [[ -d $$RESBASE/$$CSVTREE ]] ; do \
		cd $$RESBASE/$$CSVTREE; \
		if [[ n$$CSVTREE == "n" ]] ; then \
		  MYSEP=""; \
		else \
		 MYSEP="/"; \
		fi; \
		CSVTREE="$$CSVTREE""$$MYSEP"$$(ls -Art | tail -n 1); \
	done; \
	cd $$MYPROJBASE; \
	CSVTREE="$${CSVTREE%_part*}" ; \
	CSVTREE="$${CSVTREE%.csv*}" ; \
	while [[ n"$$CSVTREE" != "n" ]] ; do \
	  if ls ./results/$$CSVTREE* 1> /dev/null 2>&1; then \
		  :; \
		else \
		  echo "File patterns " ./results/$$CSVTREE* "do not exist."; \
			break; \
		fi; \
		echo rm -rf; \
		for f in ./results/$$CSVTREE* ; do \
		  echo "  " $$f; \
		done ;\
		for f in ./storage/$$CSVTREE* ; do \
		  echo "  " $$f; \
		done ;\
		read -p "?[y/n] " PERM; \
		if [[ $$PERM == "y" ]] ; then \
			echo rm -rf ./results/$$CSVTREE*; \
			rm -rf ./results/$$CSVTREE*; \
			CSVTREE=""; \
		elif [[ $$PERM == "n" ]] ; then \
		  echo "----------"; \
		  echo "Enter the config tree to remove [Leave Empty to Exit]: "; \
			echo "  Ex: 00_scratch/scratch1 "; \
			echo "  Ex: 01_firth_1layer/firth_1layer "; \
			read -p "? " CSVTREE; \
		else \
		  CSVTREE=""; \
		fi; \
	done
