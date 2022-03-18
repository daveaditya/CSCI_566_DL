rm -f assignment2.zip
zip -r assignment2.zip . -i "*.ipynb" "problem_1_solution.pdf" "problem_2_solution.pdf" -x "*.ipynb_checkpoints*"
