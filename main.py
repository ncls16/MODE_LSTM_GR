import subprocess
import time

def run_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

def run_silent_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")

def main():
    run_gr4j = input("Do you want to run main_GR4J.py? (yes/[no]): ").strip().lower()
    if run_gr4j == 'yes':
        run_script('main_GR4J.py')
    
    run_lstm = input("Do you want to run main_LSTM_training.py? (yes/[no]): ").strip().lower()
    if run_lstm == 'yes':
        variable_name = 'nom'
        variable_value = input(f"quelle partie effectuer ? (Loic, Emma, Fabio, Nicolas) {variable_name}: ").strip()

        with open('main_LSTM_training.py', 'r') as file:
            lines = file.readlines()

        with open('main_LSTM_training.py', 'w') as file:
            for line in lines:
                # Si la ligne commence par le nom de la variable, modifie la ligne
                if line.startswith(variable_name):
                    line = f"{variable_name} = '{variable_value}'\n"
                file.write(line)


        run_script('main_LSTM_training.py')
    print("the calibration is done")
    print(f"the script read_shp.py is running ...")
    run_script('read_shp.py')
    print(f"the script merge_resultats.py is running ...")
    run_silent_script('merge_resultat.py')
    print(f"the script fusion.py is running ...")
    run_script('fusion.py')
    print(f"the script plots.py is running ...")
    run_script('plots.py')

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total time taken (HMS): {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")