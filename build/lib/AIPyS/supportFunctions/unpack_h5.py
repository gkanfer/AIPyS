from AIPyS.CLI.promptParameters import check_and_prompt_parameters

def parametersInspec(option, user_parameters_path):
    parameters,non_counts,missingParam = check_and_prompt_parameters(option, user_parameters_path)
    if non_counts > 0:
        print(f" {subkey} is required. use set_parameters --{subkey} to update parameters" for subkey in missingParam)
        return
    else:
        return parameters
