#include <random>
#include <algorithm>
#include "tiramisu_code_generator.h"

//=====================================================================tiramisu_code_generator==========================================================================================================


bool inf(int i, int j) { return (i < j); }



vector<variable *> generate_variables(int nb_variables, int name_from, int *inf_values, vector<constant *> constants) {
    vector<variable *> variables;
    for (int i = name_from; i < nb_variables + name_from; ++i) {
        variables.push_back(new variable("i" + to_string(i), i, inf_values[i - name_from], constants[i - name_from]));
    }
    return variables;
}
vector<variable *> generate_variables(int nb_variables, int name_from, vector<int> inf_values, vector<constant *> constants, int vects_from) {
    vector<variable *> variables;
    for (int i = vects_from; i < nb_variables + vects_from; ++i) {
        variables.push_back(new variable("i" + to_string(i + name_from - vects_from), i + name_from, inf_values[i], constants[i]));
    }
    return variables;
}


//automatically generating a tiramisu code with convolutional layers
void generate_tiramisu_code_conv(int code_id, int nb_layers, double *padding_probs, string *default_type_tiramisu,
                                 string *default_type_wrapper) {
    //initializations
    vector<int> padding_types = generate_padding_types(nb_layers, padding_probs);
    tiramisu_code *code = new tiramisu_code("function" + to_string(code_id), &padding_types, default_type_tiramisu);

    generate_cpp_wrapper(code->function_name, code->buffers, default_type_wrapper, code_id);
    generate_h_wrapper(code->function_name, code->buffers, code_id);
}


//automatically generating a multiple computations tiramisu code with the associated wrapper
void generate_tiramisu_code_multiple_computations(int code_id, int nb_stages, double *probs,
                                                  int max_nb_dims, vector<int> scheduling_commands, bool all_schedules,
                                                  int nb_inputs, string *default_type_tiramisu,
                                                  string *default_type_wrapper, int offset,
                                                  vector<int> *tile_sizes, vector<int> *unrolling_factors,
                                                  int nb_rand_schedules, vector<dim_stats *> *stats) {


    //initializations
    srand(code_id);
    cout << "_________________code " + to_string(code_id) + "________________" << endl;
    //offset is max_offset (stencils) from input.txt
    offset = rand() % offset + 1;

    //max_nb_dims from input.txt (number of nested loops)
    int id = 0, nb, nb_shared_dims = (rand() % (max_nb_dims - 2)) + 2 , sum = 0, const_sum = MAX_MEMORY_SIZE - nb_shared_dims * MIN_LOOP_DIM;
//    int id = 0, nb, nb_dims = max_nb_dims, sum = 0, const_sum = MAX_MEMORY_SIZE - nb_dims * MIN_LOOP_DIM;

    vector<int> computation_shared_dims;

    //computation_dims contains the range of each iterator??, makes it a power of 2
    for (int i = 0; i < nb_shared_dims; ++i) {
        computation_shared_dims.push_back((rand() % MAX_CONST_VALUE) + 1);
        sum += computation_shared_dims[i];
    }

    for (int i = 0; i < nb_shared_dims; ++i) {
        computation_shared_dims[i] *= const_sum;
        computation_shared_dims[i] /= sum;
        computation_shared_dims[i] += MIN_LOOP_DIM;
        computation_shared_dims[i] = (int) pow(2.0, computation_shared_dims[i]);
    }
//    (*stats)[nb_shared_dims - 2]->data_sizes[const_sum]++;


//    (*stats)[nb_shared_dims - 2]->nb_progs++;


    string function_name = "function" + to_string(code_id);
//    vector<variable *> variables, variables_stencils;
    vector<variable *> shared_variables, shared_variables_stencils;
    vector<variable*> all_vars;
    vector<computation *> computations;
    vector<buffer *> buffers;
    vector<input *> inputs;
    vector<computation_abstract *> abs, abs1;

    int *shared_variables_min_values = new int[computation_shared_dims.size()];
    vector<constant *> shared_variable_max_values;
    int *shared_variables_min_values_stencils = new int[computation_shared_dims.size()];
    vector<constant *> shared_variable_max_values_stencils;

    nb = rand() % nb_stages + 1;
    //nb = nb_stages;
    bool st = false;

    int inp, random_index;

    map<int, vector<int>> indexes;
    vector<vector<variable *>> variables_inputs;
    vector<variable *> vars_inputs;

    vector<int> types = computation_types(nb_stages, probs), new_indices;
    computation *stage_computation;

    //creates iterators (called variables here) (for stencil and non stencil) with ranges from computation_dims
    for (int i = 0; i < computation_shared_dims.size(); ++i) {
        shared_variables_min_values[i] = 0;
        shared_variable_max_values.push_back(new constant("c" + to_string(i), computation_shared_dims[i]));
        shared_variables_min_values_stencils[i] = 0;
        shared_variable_max_values_stencils.push_back(
                new constant("c" + to_string(i) + " - " + to_string(offset)+ " - " + to_string(offset),
                computation_shared_dims[i] - offset - offset));
    }

    shared_variables=generate_variables(computation_shared_dims.size(), 0, shared_variables_min_values, shared_variable_max_values);
    shared_variables_stencils=generate_variables(computation_shared_dims.size(),  100,
                                                 shared_variables_min_values_stencils, shared_variable_max_values_stencils);












    vector<vector<int>> computation_unshared_dims;

    int nb_new_dims=0 ;
    vector<vector<int>> unshared_variables_min_values;
    vector<vector<int>> unshared_variables_min_values_stencils;
    vector<vector<constant *>> unshared_variable_max_values;
    vector<vector<constant *>> unshared_variable_max_values_stencils;
    vector<vector <variable *>> unshared_variables, unshared_variables_stencils;

    //generate randomly nb(number of computations) list of variables and computation_dims arrays
    for (int cpt_compt = 0; cpt_compt<nb; cpt_compt++){

        int reused_depth = 0 ;
        double reuse_var = (double) rand() / (RAND_MAX);
        if ((reuse_var < ITERATOR_REUSE_PROB) && (cpt_compt>0)){
            int reused_branch = rand() % cpt_compt;
            if (cpt_compt == 1)
                reused_branch = 0; // if it's generating the second computation (comp 1 ) only branch 0 exits
            if (computation_unshared_dims[reused_branch].size()<=1){
                reused_depth = 1;
            } else {
                reused_depth = (rand() % (computation_unshared_dims[reused_branch].size())) +1;
            }
            computation_unshared_dims.push_back(vector<int> (computation_unshared_dims[reused_branch].begin(), computation_unshared_dims[reused_branch].begin() + reused_depth));
            unshared_variable_max_values.push_back(vector<constant *> (unshared_variable_max_values[reused_branch].begin(), unshared_variable_max_values[reused_branch].begin() + reused_depth));
            unshared_variable_max_values_stencils.push_back(vector<constant *> (unshared_variable_max_values_stencils[reused_branch].begin(), unshared_variable_max_values_stencils[reused_branch].begin() + reused_depth));
            unshared_variables_min_values.push_back(vector<int> (unshared_variables_min_values[reused_branch].begin(), unshared_variables_min_values[reused_branch].begin() + reused_depth ));
            unshared_variables_min_values_stencils.push_back(vector<int> (unshared_variables_min_values_stencils[reused_branch].begin(), unshared_variables_min_values_stencils[reused_branch].begin() + reused_depth ));
            unshared_variables.push_back(vector<variable *> (unshared_variables[reused_branch].begin(), unshared_variables[reused_branch].begin() + reused_depth));
            unshared_variables_stencils.push_back(vector<variable *> (unshared_variables_stencils[reused_branch].begin(), unshared_variables_stencils[reused_branch].begin() + reused_depth));

            nb_new_dims = (rand() % (max_nb_dims - nb_shared_dims + 1 - reused_depth))  ;
        } else {
            nb_new_dims = (rand() % (max_nb_dims - nb_shared_dims + 1)) ;
            if (nb_new_dims==0) nb_new_dims=1; // this is because the code doesn't yet support stencils on shared iterator, so there must always be at least one unshared dim, remove this line once it supports that

            vector<int> place_holder_int;
            computation_unshared_dims.push_back(place_holder_int);
            vector<constant *> place_holder;
            unshared_variable_max_values.push_back(place_holder);
            vector<constant *> place_holder_stencil;
            unshared_variable_max_values_stencils.push_back(place_holder_stencil);
            vector<int> place_holder_minval;
            unshared_variables_min_values.push_back(place_holder_minval);
            vector<int> place_holder_minval_stencil;
            unshared_variables_min_values_stencils.push_back(place_holder_minval_stencil);
            vector <variable *> place_holder_var;
            unshared_variables.push_back(place_holder_var);
            vector <variable *> place_holder_var_stencil;
            unshared_variables_stencils.push_back(place_holder_var_stencil);


        }




        for (int i = reused_depth; i < nb_new_dims + reused_depth; ++i) {
            computation_unshared_dims[cpt_compt].push_back((rand() % MAX_CONST_VALUE) + 1);
            sum += computation_unshared_dims[cpt_compt][i];
        }
        for (int i = reused_depth; i < nb_new_dims + reused_depth; ++i) {
            computation_unshared_dims[cpt_compt][i] *= const_sum;
            computation_unshared_dims[cpt_compt][i] /= sum;
            computation_unshared_dims[cpt_compt][i] += MIN_LOOP_DIM;
            computation_unshared_dims[cpt_compt][i] = (int) pow(2.0, computation_unshared_dims[cpt_compt][i]);
        }

        for (int i = reused_depth; i < nb_new_dims + reused_depth; ++i) {
            unshared_variables_min_values[cpt_compt].push_back(0);
            unshared_variable_max_values[cpt_compt].push_back(new constant("c" + to_string(cpt_compt) + to_string(i), computation_unshared_dims[cpt_compt][i]));
            unshared_variables_min_values_stencils[cpt_compt].push_back(0);
            unshared_variable_max_values_stencils[cpt_compt].push_back(
                    new constant("c" + to_string(cpt_compt) + to_string(i) + " - " + to_string(offset)+ " - " + to_string(offset),
                                 computation_unshared_dims[cpt_compt][i] - offset - offset));
        }

        vector <variable *> new_vars = generate_variables(nb_new_dims, 1000 + cpt_compt*10,
                           unshared_variables_min_values[cpt_compt], unshared_variable_max_values[cpt_compt], reused_depth);
        unshared_variables[cpt_compt].insert(unshared_variables[cpt_compt].end(), new_vars.begin(), new_vars.end());
        vector <variable *> new_vars_stencil = generate_variables(nb_new_dims,  1100 + cpt_compt*10,
                                                                  unshared_variables_min_values_stencils[cpt_compt], unshared_variable_max_values_stencils[cpt_compt], reused_depth);
        unshared_variables_stencils[cpt_compt].insert(unshared_variables_stencils[cpt_compt].end(), new_vars_stencil.begin(), new_vars_stencil.end() );

    }



//    (*stats)[nb_shared_dims - 2]->data_sizes[const_sum]++;
//    (*stats)[nb_shared_dims - 2]->nb_progs++;
//    vector<variable *> variables, variables_stencils;
//    vector<variable *> shared_variables, shared_variables_stencils;
//    vector<variable*> all_vars;
//    vector<computation *> computations;
//    vector<buffer *> buffers;
//    vector<input *> inputs;
//    vector<computation_abstract *> abs, abs1;

//    map<int, vector<int>> indexes;
//    vector<vector<variable *>> variables_inputs;
//    vector<variable *> vars_inputs;

//    vector<int> types = computation_types(nb_stages, probs), new_indices;
//    computation *stage_computation;

    //creates iterators (called variables here) (for stencil and non stencil) with ranges from computation_dims



//    vector<variable *> variables_indep=generate_variables(max_nb_dims-computation_dims.size(), 1000, variables_min_values, variable_max_values);





    // since all computations must have same iterators, if one comp is a stencil all the other comps will have stencil iterators

//    if(find(types.begin(), types.end(), STENCIL) != types.end()) {
//
//        variables = variables_stencils;
//
//    }
//
//    vector<variable *> conc_var;
//    conc_var.insert( conc_var.end(), variables.begin(), variables.end() );
//    conc_var.insert( conc_var.end(), variables_indep.begin(), variables_indep.end() );
//    variables = conc_var;


    // variables becomes shared variables, create a list of list of indep variables randomly (intersection can be not empty but the order
    // must be respected), each time generating a comp use as variable parameter one of the indep variable list concatenated with the shared var list
    vector<variable *> variables;
    vector<variable *> variables_stencils;
    vector<int> computation_dims;
    for (int i = 0; i < nb; ++i) {
        // nb_inputs is max number of input buffers
        int comp_variables_choice = rand() % nb;
        variables.clear();
        variables_stencils.clear();
        computation_dims.clear();
        computation_dims.insert(computation_dims.begin(), computation_shared_dims.begin(), computation_shared_dims.end());
        computation_dims.insert(computation_dims.begin(), computation_unshared_dims[comp_variables_choice].begin(), computation_unshared_dims[comp_variables_choice].end());


        inp = rand() % nb_inputs + 1;
        switch (types[i]) {
            case ASSIGNMENT:
                variables.insert(variables.end(), shared_variables.begin(), shared_variables.end());
                variables.insert(variables.end(), unshared_variables[comp_variables_choice].begin(), unshared_variables[comp_variables_choice].end() );

                stage_computation = generate_computation("comp" + to_string(i), variables, ASSIGNMENT, {}, {}, 0,
                                                         *default_type_tiramisu, id++);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), computation_dims, OUTPUT_BUFFER, &abs));
                break;
            case ASSIGNMENT_INPUTS:
                variables.insert(variables.end(), shared_variables.begin(), shared_variables.end());
                variables.insert(variables.end(), unshared_variables[comp_variables_choice].begin(), unshared_variables[comp_variables_choice].end() );
                abs1.clear();
                variables_inputs.clear();
                for (int j = 0; j < inp; ++j) {
                    int nb_vars = rand() % (variables.size() - 1) + 1;
                    vars_inputs.clear();
                    if ((double) rand() / (RAND_MAX) < SHUFFLE_SAME_SIZE_PROB) {
                        indexes = indexes_by_size(variables);
                        for (int k = 0; k < variables.size(); ++k) {
                            auto var = indexes.find(variables[k]->sup_value->value)->second;
                            random_index = rand() % var.size();
                            vars_inputs.push_back(variables[var[random_index]]);
                            indexes.find(variables[k]->sup_value->value)->second.erase(
                                    indexes.find(variables[k]->sup_value->value)->second.begin() + random_index);
                        }
                    }
                    else {
                        vars_inputs = variables;
                    }
                    int n_erase = variables.size() - nb_vars;
                    vector<int> input_dims = computation_dims;
                    for (int l = 0; l < n_erase; ++l) {
                        random_index = rand() % vars_inputs.size();
                        vars_inputs.erase(vars_inputs.begin() + random_index);
                        input_dims.erase(input_dims.begin() + random_index);
                    }
                    variables_inputs.push_back(vars_inputs);


                    input *in = new input("input" + to_string(i) + to_string(j), &(variables_inputs[j]),
                                          *default_type_tiramisu, id++);
                    inputs.push_back(in);
                    abs = {in};
                    buffers.push_back(
                            new buffer("buf" + to_string(i) + to_string(j), input_dims, INPUT_BUFFER, &abs));
                    abs1.push_back(in);
                }
                //TODO: use previous stages as inputs in abs1
                stage_computation = generate_computation("comp" + to_string(i), variables, ASSIGNMENT_INPUTS, abs1, {},
                                                         0, *default_type_tiramisu, id++);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), computation_dims, OUTPUT_BUFFER, &abs));


                break;
            case STENCIL:
                variables_stencils.insert(variables_stencils.end(), shared_variables.begin(), shared_variables.end());
                variables_stencils.insert(variables_stencils.end(), unshared_variables_stencils[comp_variables_choice].begin(), unshared_variables_stencils[comp_variables_choice].end() );
                vector <variable*> stencils_input_variables;
                stencils_input_variables.insert(stencils_input_variables.end(), shared_variables.begin(), shared_variables.end());
                stencils_input_variables.insert(stencils_input_variables.end(),unshared_variables[comp_variables_choice].begin(), unshared_variables[comp_variables_choice].end());

                vector<int> var_nums_copy;
                vector<int> var_nums_copy_copy;

                // TODO: I actually use only unshared variables for stencil operations (as stencil iterators)
                for (int l = shared_variables.size(); l < variables_stencils.size(); ++l) {
                    var_nums_copy_copy.push_back(l);
                }

                auto rng = default_random_engine{};
                shuffle(begin(var_nums_copy_copy), end(var_nums_copy_copy), rng);


                int stencil_size;
                (variables_stencils.size() > MAX_STENCIL_SIZE) ? stencil_size = MAX_STENCIL_SIZE
                                                               : stencil_size = variables_stencils.size();

                for (int i = 0; i < rand() % stencil_size + 1; ++i) {
                    var_nums_copy.push_back(var_nums_copy_copy[i]);
                }

                sort(var_nums_copy.begin(), var_nums_copy.end(), inf);

                st = true;
                //clearing abs for not using the previous computation as input
                abs.clear();
                if (abs.empty()) {
                    // the "0" is because only one input is generated for the stencil
                    input *in = new input("input" + to_string(i) + "0", &(stencils_input_variables), *default_type_tiramisu, id++);
                    abs = {in};
                    inputs.push_back(in);
                    abs = {in};
                    // the "0" is because only one input is generated for the stencil
                    buffers.push_back(
                            new buffer("buf" + to_string(i) + "0", computation_dims, INPUT_BUFFER, &abs));
                    abs1.push_back(in);
                }
                if (code_id==1007)
                    cout<< "1007";
                stage_computation = generate_computation("comp" + to_string(i), variables_stencils, STENCIL, abs,
                                                         var_nums_copy, offset, *default_type_tiramisu, id++);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), computation_dims, OUTPUT_BUFFER, &abs));
                break;
        }
//        (*stats)[computation_dims.size() - 2]->types[types[i]]->nb_assignments++;
    }

   /* for (int k = 0; (k < variables_stencils.size()) && st; ++k) {
        variables.push_back(variables_stencils[k]);
    }*/

//    for (int k = 0; (k < stencils_input_variables.size()) && st; ++k) {
//        variables.push_back(stencils_input_variables[k]);
//    }

//    for (int j = 0; j < shared_variables.size(); ++j) {
//            all_vars.push_back(shared_variables[j]);
//    }
//    for (int k= 0 ; k<unshared_variables.size(); k++){
//        for ( int j =0; j<unshared_variables[k].size(); j++){
//            all_vars.push_back(unshared_variables[k][j]);
//            all_vars.push_back(unshared_variables_stencils[k][j]);
//        }
//    }

// parcourir toutes les var utilisé uniquement pour les ajouter a all vars


    for (int j = 0; j < inputs.size(); ++j) {
        for (int k = 0; k< inputs[j]->variables.size(); k++){
            if (!(contains(all_vars, inputs[j]->variables[k]))){
                all_vars.push_back(inputs[j]->variables[k]);
            }
        }

    }

    for (int j = 0; j < computations.size(); ++j) {
        for (int k = 0; k< computations[j]->variables.size(); k++){
            if (!(contains(all_vars, computations[j]->variables[k]))){
                all_vars.push_back(computations[j]->variables[k]);
            }
        }

    }


    vector<schedule *> schedules;

    node_class *n = comps_to_node(computations, code_id);
    generate_json_one_node(n, code_id);

    vector<schedule_params> schedules_parameters;
    for (int i = 0; i < scheduling_commands.size(); ++i) {
        switch(scheduling_commands[i]){
            case INTERCHANGE:
                schedules_parameters.push_back({INTERCHANGE, {}, 1});
                break;
            case TILING:
                schedules_parameters.push_back({TILING, *tile_sizes, 1});
                break;
            case UNROLL:
                schedules_parameters.push_back({UNROLL, *unrolling_factors, 1});
                break;
            default:
                cout << "The command should be either interchange, tiling or unrolling." << endl;
                return;
        }
    }


    vector<vector<schedule*>> schedules_exhaustive = {};
    vector<vector<variable*>> variables_exhaustive;
    vector<schedules_class*> schedule_classes;

    vector<variable*> all_schedule_variables;

    if (all_schedules) {
        //generate_all_schedules(schedules_parameters, computations[0], &schedules_exhaustive, &variables_exhaustive,
        //                       &schedule_classes);
//        generate_all_schedules_multiple_adj_comps(schedules_parameters, computations, &schedules_exhaustive, &variables_exhaustive,
//                                &schedule_classes);
        generate_all_schedules_shared_vars(schedules_parameters, computations, &schedules_exhaustive, &variables_exhaustive,
                                &schedule_classes, shared_variables);
    }
    else {
       // generate_random_schedules(nb_rand_schedules, schedules_parameters, computations[0], &schedules_exhaustive,
          //                        &variables_exhaustive, &schedule_classes);
          //TODO gen schedules not implemented for the general case
        generate_random_schedules_multiple_adj_comps(nb_rand_schedules, schedules_parameters, computations, &schedules_exhaustive,
                                                     &variables_exhaustive, &schedule_classes);
    }


    tiramisu_code *code;

    vector<constant *> variable_max_values;
//    variable_max_values.insert(variable_max_values.end(), shared_variable_max_values.begin(), shared_variable_max_values.end());
////    variable_max_values.insert(variable_max_values.end(), shared_variable_max_values_stencils.begin(), shared_variable_max_values_stencils.end());
//
//    for (int i = 0; i<unshared_variable_max_values.size(); i++){
//        variable_max_values.insert(variable_max_values.end(), unshared_variable_max_values[i].begin(), unshared_variable_max_values[i].end());
////        variable_max_values.insert(variable_max_values.end(), unshared_variable_max_values_stencils[i].begin(), unshared_variable_max_values_stencils[i].end());
//    }


    for (int j = 0; j < all_vars.size(); ++j) {
        if (!(contains(variable_max_values, all_vars[j]->sup_value))){
            variable_max_values.push_back(all_vars[j]->sup_value);
        }
    }



    for (int i = 0; i < variables_exhaustive.size(); ++i) {
        all_schedule_variables = all_vars;
        for (int j = 0; j < variables_exhaustive[i].size(); ++j) {
            if (!(contains(all_schedule_variables, variables_exhaustive[i][j]))){
                all_schedule_variables.push_back(variables_exhaustive[i][j]);
            }
        }
       // all_schedule_variables.push_back(new variable("i_vec", 21));
       // all_schedule_variables.push_back(new variable("i_vec1", 22));
        //schedules_exhaustive[i].push_back(new schedule({computations[0], computations[1]}, AFTER, {0}, {}));
        if (i != 0) {
            code = new tiramisu_code(code_id,
                                     function_name + "_schedule_" +
                                     to_string(i -1),
                                     &computations, &all_schedule_variables,
                                     &variable_max_values, &inputs, &buffers,
                                     default_type_tiramisu, &schedules_exhaustive[i]);
        }
        else{
            code = new tiramisu_code(code_id,
                                     function_name + "_no_schedule",
                                     &computations, &all_schedule_variables,
                                     &variable_max_values, &inputs, &buffers,
                                     default_type_tiramisu, &schedules_exhaustive[i]);
        }
        generate_cpp_wrapper(code->function_name, buffers, default_type_wrapper, code_id);
        generate_h_wrapper(code->function_name, buffers, code_id);
        generate_json_schedules(schedule_classes[i], code_id, code->function_name);
    }



}



/*node_class *comp_to_node(computation *comp, int seed) {

    vector<iterator_class *> iterators_array;
    for (int i = 0; i < comp->variables.size(); ++i) {
        iterators_array.push_back(new iterator_class(comp->variables[i]->id, comp->variables[i]->inf_value,
                                                     comp->variables[i]->sup_value->value));
    }

    iterators_class *iterators = new iterators_class(iterators_array.size(), iterators_array);

    vector<mem_access_class *> mem_accesses_array;
    for (int i = 0; i < comp->used_comps.size(); ++i) {
        mem_accesses_array.push_back(new mem_access_class(comp->used_comps[i]->id, comp->used_comps_accesses[i]));
    }

    mem_accesses_class *mem_accesses = new mem_accesses_class(mem_accesses_array.size(), mem_accesses_array);

    vector<int> its, its_stencils;

    for (int i = 0; i < iterators_array.size(); ++i) {
        its.push_back(iterators_array[i]->id);
    }

    vector<input_class *> inputs;
    //not flexible, based on the fact that only stencils use different iterators
    if (comp->type == STENCIL) {
        for (int i = 0; i < comp->used_comps[0]->variables.size(); ++i) {
            iterators->n++;
            iterators->it_array.push_back(new iterator_class(comp->used_comps[0]->variables[i]->id,
                                                             comp->used_comps[0]->variables[i]->inf_value,
                                                             comp->used_comps[0]->variables[i]->sup_value->value));
            its_stencils.push_back(comp->used_comps[0]->variables[i]->id);
        }
        inputs.push_back(new input_class(comp->used_comps[0]->id, comp->used_comps[0]->data_type, its_stencils));
    } else {
        for (int i = 0; i < comp->used_comps.size(); ++i) {
            inputs.push_back(new input_class(comp->used_comps[i]->id, comp->used_comps[i]->data_type, its));
        }
    }

    inputs_class *all_inputs = new inputs_class(inputs.size(), inputs);


    computation_class *computation = new computation_class(comp->id, comp->data_type, its, comp->op_stats,
                                                           mem_accesses);

    computations_class *computations = new computations_class(1, {computation});

    assignment_class *assignment = new assignment_class(comp->id, 0);

    assignments_class *assignments = new assignments_class(1, {assignment});

    vector<loop_class *> loops_array;
    for (int i = 0; i < comp->variables.size(); ++i) {
        loops_array.push_back(new loop_class(i, i - 1, 0, iterators_array[i]->id, new assignments_class(0, {})));
    }
    loops_array[comp->variables.size() - 1]->assignments = assignments;

    loops_class *loops = new loops_class(comp->variables.size(), loops_array);

    node_class *node = new node_class(seed, loops, computations, iterators, all_inputs);

    node->code_type = comp->type;

    return node;
}*/

node_class *comps_to_node(vector<computation*> comps, int seed) {


//    vector<iterator_class *> iterators_array;
//    for (int i = 0; i < comps[0]->variables.size(); ++i) {
//        iterators_array.push_back(new iterator_class(comps[0]->variables[i]->id, comps[0]->variables[i]->inf_value,
//                                                     comps[0]->variables[i]->sup_value->value));
//    }
//
//    iterators_class *iterators = new iterators_class(iterators_array.size(), iterators_array);


    vector<iterator_class *> comps_iterators_array;
    vector<int> comps_iterators_ids_array;
    vector<iterator_class *> inputs_iterators_array;
    vector<int> inputs_iterators_ids_array;

    for (int i=0; i<comps.size(); i++){
        for( int j=0 ; j< comps[i]->variables.size(); j++){

            if(find(comps_iterators_ids_array.begin(), comps_iterators_ids_array.end(), comps[i]->variables[j]->id) == comps_iterators_ids_array.end()){
                comps_iterators_ids_array.push_back(comps[i]->variables[j]->id);
                comps_iterators_array.push_back(new iterator_class(comps[i]->variables[j]->id, comps[i]->variables[j]->inf_value,
                                                                   comps[i]->variables[j]->sup_value->value));
            }
        }
        for (int j=0; j< comps[i]->used_comps.size(); j++){
            for (int k= 0; k < comps[i]->used_comps[j]->variables.size(); k++){
                if(find(inputs_iterators_ids_array.begin(), inputs_iterators_ids_array.end(), comps[i]->used_comps[j]->variables[k]->id) == inputs_iterators_ids_array.end()){
                    inputs_iterators_ids_array.push_back(comps[i]->used_comps[j]->variables[k]->id);
                    inputs_iterators_array.push_back(new iterator_class(comps[i]->used_comps[j]->variables[k]->id, comps[i]->used_comps[j]->variables[k]->inf_value,
                                                                        comps[i]->used_comps[j]->variables[k]->sup_value->value));
                }
            }
        }
    }

    vector<iterator_class *> all_iterators_array;
    for (int i=0; i< comps_iterators_array.size(); i++){
        all_iterators_array.push_back(comps_iterators_array[i]);
    }
    for (int i=0; i< inputs_iterators_array.size(); i++){
        if( find(comps_iterators_ids_array.begin(), comps_iterators_ids_array.end(), inputs_iterators_array[i]->id) == comps_iterators_ids_array.end()){
            all_iterators_array.push_back(inputs_iterators_array[i]);
        }
    }
    iterators_class *iterators = new iterators_class(all_iterators_array.size(), all_iterators_array);


    /*vector<mem_access_class *> mem_accesses_array;
    for (int i = 0; i < comps[0]->used_comps.size(); ++i) {
        mem_accesses_array.push_back(new mem_access_class(comps[0]->used_comps[i]->id, comps[0]->used_comps_accesses[i]));
    }

    mem_accesses_class *mem_accesses = new mem_accesses_class(mem_accesses_array.size(), mem_accesses_array);*/



    vector<mem_accesses_class*> mem_accesses;
    for (int j=0; j<comps.size(); j++){
        vector<mem_access_class *> mem_accesses_array;
        for (int i = 0; i < comps[j]->used_comps.size(); ++i) {
            mem_accesses_array.push_back(new mem_access_class(comps[j]->used_comps[i]->id, comps[j]->used_comps_accesses[i]));
        }
        mem_accesses.push_back(new mem_accesses_class(mem_accesses_array.size(), mem_accesses_array));
    }


    //vector<int> its, its_stencils;

//    for (int i = 0; i < iterators_array.size(); ++i) {
//        its.push_back(iterators_array[i]->id);
//    }


//    //not flexible, based on the fact that only stencils use different iterators
//    if (comps[0]->type == STENCIL) {
//        for (int i = 0; i < comps[0]->used_comps[0]->variables.size(); ++i) {
//            iterators->n++;
//            iterators->it_array.push_back(new iterator_class(comps[0]->used_comps[0]->variables[i]->id,
//                                                             comps[0]->used_comps[0]->variables[i]->inf_value,
//                                                             comps[0]->used_comps[0]->variables[i]->sup_value->value));
//            its_stencils.push_back(comps[0]->used_comps[0]->variables[i]->id);
//        }
//       // inputs.push_back(new input_class(comps[0]->used_comps[0]->id, comps[0]->used_comps[0]->data_type, its_stencils));
//    } else {
//        for (int i = 0; i < comps[0]->used_comps.size(); ++i) {
//            inputs.push_back(new input_class(comps[0]->used_comps[i]->id, comps[0]->used_comps[i]->data_type, its));
//        }
//
//    }

    vector<input_class *> inputs;
    vector<int> inputs_ids;
    for (int j = 0; j < comps.size(); j++){
        for (int i = 0; i < comps[j]->used_comps.size(); ++i) {
            //cout << comps[j]->used_comps[i]->id;
            //insert the input in the vector if it's not already there, and it's not a computation
            if((find(inputs_ids.begin(), inputs_ids.end(), comps[j]->used_comps[i]->id) == inputs_ids.end()) && (comps[j]->used_comps[i]->name.find("comp") == string::npos)) {
                vector <int> input_iterators;
                for (int k= 0; k < comps[j]->used_comps[i]->variables.size(); k++){
                    input_iterators.push_back(comps[j]->used_comps[i]->variables[k]->id);
                }
                inputs.push_back(new input_class(comps[j]->used_comps[i]->id, comps[j]->used_comps[i]->data_type, input_iterators));
                inputs_ids.push_back(comps[j]->used_comps[i]->id);

            }
        }
    }

    inputs_class *all_inputs = new inputs_class(inputs.size(), inputs);


    vector<computation_class*> computation;
    for (int i=0; i<comps.size(); i++){
        vector <int> comp_iterators_ids;
        for( int j=0 ; j< comps[i]->variables.size(); j++){
            comp_iterators_ids.push_back(comps[i]->variables[j]->id);
        }
        computation.push_back(new computation_class(comps[i]->id, comps[i]->data_type, comp_iterators_ids, comps[i]->op_stats,
                                                    mem_accesses[i]));

    }

    //computation_class *computation = new computation_class(comp->id, comp->data_type, its, comp->op_stats,mem_accesses);

    computations_class *computations = new computations_class(computation.size(), computation);

    //assignment_class *assignment = new assignment_class(comps[0]->id, 0);
    vector<assignment_class*> assignment;
    for (int i=0; i<comps.size();i++){
        assignment.push_back(new assignment_class(comps[i]->id, i));
    }

    assignments_class *assignments = new assignments_class(comps.size(), assignment);


    vector<loop_class *> loops_array;
    for (int i = 0; i < comps[0]->variables.size(); ++i) {
        loops_array.push_back(new loop_class(i, i - 1, 0, comps[0]->variables[i]->id, new assignments_class(0, {})));
        //loops_array.push_back(new loop_class(i, i - 1, 0, iterators_array[i]->id, new assignments_class(0, {})));
    }
    loops_array[comps[0]->variables.size() - 1]->assignments = assignments;

    loops_class *loops = new loops_class(comps[0]->variables.size(), loops_array);

    node_class *node = new node_class(seed, loops, computations, iterators, all_inputs);

    node->code_type = comps[0]->type;

    return node;
}

//=====================================================================random_generator==========================================================================================================
//automatically generating computation
computation *generate_computation(string name, vector<variable *> computation_variables, int computation_type,
                                  vector<computation_abstract *> inputs, vector<int> var_nums, int offset,
                                  string data_type, int id) {
    computation *c = new computation(name, computation_type, &computation_variables, data_type, id);
    vector<int> vect;
    vector<vector<int>> access;
    if (computation_type == ASSIGNMENT) {
        c->expression = assignment_expression_generator(c->op_stats);
        c->used_comps = {};
    }

    if (computation_type == ASSIGNMENT_INPUTS) {
        c->expression = assignment_expression_generator_inputs(inputs, c->op_stats);
        c->used_comps = inputs;
        c->used_comps_accesses.clear();
        for (int i = 0; i < inputs.size(); ++i) {
            access.clear();
            for (int j = 0; j < c->used_comps[i]->variables.size(); ++j) {
                vect.clear();
                for (int k = 0; k < c->variables.size() + 1; ++k) {
                    if (k < c->variables.size() && (c->used_comps[i]->variables[j]->id == c->variables[k]->id)) {
                        vect.push_back(1);
                    } else
                        vect.push_back(0);
                }
                access.push_back(vect);
            }
            c->used_comps_accesses.push_back(access);
        }
    }

    if (computation_type == STENCIL) {
        c->used_comps_accesses.clear();
        c->used_comps.clear();
        for (int i = 0; i < offset * (pow(3.0, var_nums.size()) - 1) + 1; ++i) {
            access.clear();
            for (int j = 0; j < c->variables.size(); ++j) {
                vect.clear();
                for (int k = 0; k < c->variables.size() + 1; ++k) {
                    if (j == k) {
                        vect.push_back(1);
                    } else vect.push_back(0);
                }
                access.push_back(vect);
            }
            c->used_comps_accesses.push_back(access);

            c->used_comps.push_back(inputs[0]);
        }
        c->expression = stencil_expression_generator(inputs[0]->name, c, &var_nums, offset, c->op_stats,
                                                     &c->used_comps_accesses);
        vector <int> duplicate_indexes;
        for (int i = 0; i < c->used_comps_accesses.size(); i++ ){
            for (int j = i+1; j < c->used_comps_accesses.size(); j++){
                bool is_duplicate= true;
                if (c->used_comps_accesses[i].size() != c->used_comps_accesses[j].size()){
                    is_duplicate = false;
                } else {
                    for (int k = 0; k < c->used_comps_accesses[i].size(); k++){
                        if (c->used_comps_accesses[i][k] != c->used_comps_accesses[j][k]){

                            is_duplicate= false;
                        }
                    }
                }
                if (is_duplicate)
                    duplicate_indexes.push_back(j);
            }
        }
        sort(duplicate_indexes.begin(), duplicate_indexes.end(), greater<int>());
        auto last = unique(duplicate_indexes.begin(), duplicate_indexes.end());
        duplicate_indexes.erase(last, duplicate_indexes.end());
        for (int i = 0; i< duplicate_indexes.size();i++){
            c->used_comps_accesses.erase(c->used_comps_accesses.begin()+ duplicate_indexes[i]);
            c->used_comps.erase(c->used_comps.begin()+ duplicate_indexes[i]);

        }

    }

    return c;
}


//automatically generating computation expression in case of a simple assignment
string assignment_expression_generator(int **op_stats) {
    string expr = to_string(rand() % MAX_ASSIGNMENT_VAL);
    for (int i = 0; i < rand() % (MAX_NB_OPERATIONS_ASSIGNMENT - 1) + 1; ++i) {
        int op = rand() % 3;
        switch (op) {
            case 0:
                expr += " + " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                op_stats[0][0]++;
                break;
            case 1:
                expr += " - " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                op_stats[0][1]++;
                break;
            case 2:
                expr += " * " + to_string(rand() % MAX_ASSIGNMENT_VAL);
                op_stats[0][2]++;
                break;
                //       case 3:
                //         expr += " / " + to_string(rand() % (MAX_ASSIGNMENT_VAL - 1) + 1);
                //         op_stats[3]++;
                //       break;

        }
    }
    return expr;
}


//automatically generating computation expression in case of an assignment using other computations
string assignment_expression_generator_inputs(vector<computation_abstract *> inputs, int **op_stats) {
    string vars = inputs[0]->vars_to_string();
    string expr = inputs[0]->name + vars;
    for (int i = 1; i < inputs.size(); ++i) {
        int op = rand() % 3;
        switch (op) {
            case 0:
                expr += " + " + inputs[i]->name + inputs[i]->vars_to_string();
                op_stats[0][0]++;
                break;
            case 1:
                expr += " - " + inputs[i]->name + inputs[i]->vars_to_string();
                op_stats[0][1]++;
                break;
            case 2:
                expr += " * " + inputs[i]->name + inputs[i]->vars_to_string();
                op_stats[0][2]++;
                break;
                // case 3:
                //   expr += " / " + inputs[i]->name + vars + " + 1";
                //   op_stats[3]++;
                // break;

        }
    }
    return expr;
}


//automatically generating computation expression in case of stencils
string stencil_expression_generator(string input_name, computation_abstract *in, vector<int> *var_nums, int offset,
                                    int **op_stats, vector<vector<vector<int>>> *accesses) {
    vector<string> vars = in->for_stencils(*var_nums, offset, accesses);
    string expr = "(";
    for (int i = 0; i < vars.size() - 1; ++i) {
        expr += input_name + vars[i];
        if (rand() % 2) {
            expr += " + ";
            op_stats[0][0]++;
        } else {
            expr += " - ";
            op_stats[0][1]++;
        }
    }
    expr += input_name + vars[vars.size() - 1] + ")";

    return expr;
}


//returns a vector of computation types according to the probability of their occurence (used in multiple computations codes)
vector<int> computation_types(int nb_comps, double *probs) {
    vector<int> types;
    double num;
    for (int i = 0; i < nb_comps; ++i) {
        num = (double) rand() / (RAND_MAX);
        if (num < probs[0]) {
            types.push_back(ASSIGNMENT);
        } else if (num < probs[0] + probs[1]) {
            types.push_back(ASSIGNMENT_INPUTS);
        } else {
            types.push_back(STENCIL);
        }
    }
    return types;

}


//returns a vector of padding types according to the probability of their occurence (used in convolutions)
vector<int> generate_padding_types(int nb_layers, double *padding_probs) {
    vector<int> types;
    double num;
    for (int i = 0; i < nb_layers; ++i) {
        num = (double) rand() / (RAND_MAX);
        if (num < padding_probs[0]) {
            types.push_back(SAME);
        } else {
            types.push_back(VALID);
        }
    }
    return types;
}

//=============================================Managing schedules=====================================================================



vector<configuration> generate_configurations(int schedule, vector<int> factors, vector<variable*> variables, int computation_type){
    //variables : all computation variables before applying schedule
    //in_variables : variables used in schedule
    //out_variables : all computation variables after applying schedule

    vector<configuration> configs;
    variable *v1, *v2, *v3, *v4, *v5, *v6;
    configuration conf;
    conf.computation_type = computation_type;
    conf.out_variables = variables;
    switch(schedule){
        case INTERCHANGE:{
            conf.schedule = NONE;
            configs.push_back(conf);
            conf.schedule = INTERCHANGE;
            for(int i = 0; i <variables.size(); i++){
                for(int j = i + 1; j < variables.size(); j++){
                    conf.in_variables = {variables[i], variables[j]};
                    conf.out_variables = variables;
                    iter_swap(conf.out_variables.begin() + i, conf.out_variables.begin() + j);
                    configs.push_back(conf);
                }
            }
            break;
        }
        case TILING:{
            v1 = new variable("i01", 11);
            v2 = new variable("i02", 12);
            v3 = new variable("i03", 13);
            v4 = new variable("i04", 14);
            conf.schedule = NONE;
            configs.push_back(conf);
            conf.schedule = TILE_2;
            for(int i = 0; i <factors.size(); i++){
                for(int j = 0; j <factors.size(); j++){
                    for(int l = 0; l <variables.size() - 1; l++){
                        conf.factors = {factors[i], factors[j]};
                        conf.in_variables = {variables[l], variables[l + 1], v1, v2, v3, v4};
                        conf.out_variables = variables;
                        conf.out_variables.insert(conf.out_variables.begin() + l, v1);
                        conf.out_variables.insert(conf.out_variables.begin() + l + 1, v2);
                        conf.out_variables.insert(conf.out_variables.begin() + l + 2, v3);
                        conf.out_variables.insert(conf.out_variables.begin() + l + 3, v4);
                        conf.out_variables.erase(conf.out_variables.begin() + l + 4, conf.out_variables.begin() + l + 6);
                        configs.push_back(conf);
                    }
                }
            }
            if (variables.size() > 2){
                conf.schedule = TILE_3;
                v5 = new variable("i05", 15);
                v6 = new variable("i06", 16);
                for(int i = 0; i <factors.size(); i++){
                    for(int j = 0; j <factors.size(); j++){
                        for(int k = 0; k <factors.size(); k++){
                            for(int l = 0; l <variables.size() - 2; l++){
                                conf.factors = {factors[i], factors[j], factors[k]};
                                conf.in_variables = {variables[l], variables[l + 1], variables[l + 2], v1, v2, v3, v4, v5, v6};
                                conf.out_variables = variables;
                                conf.out_variables.insert(conf.out_variables.begin() + l, v1);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 1, v2);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 2, v3);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 3, v4);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 4, v5);
                                conf.out_variables.insert(conf.out_variables.begin() + l + 5, v6);
                                conf.out_variables.erase(conf.out_variables.begin() + l + 6, conf.out_variables.begin() + l + 9);
                                configs.push_back(conf);
                            }
                        }
                    }
                }
            }
            break;
        }
        case UNROLL:{
            conf.schedule = NONE_UNROLL;
            configs.push_back(conf);
            conf.schedule = UNROLL;
            conf.in_variables = {variables.back()};
            for(int i = 0; i <factors.size(); i++){
                conf.factors = {factors[i]};
                configs.push_back(conf);
            }
            break;
        }
    }
    return configs;
}



//possible ways of generating schedules :
//*exhaustively(list of (schedule, list of parameters)) :
//explore all combinations with all possible factors
//*randomly(number of schedules, list of(schedule, list of parameters, probability for schedule))

void generate_all_schedules(vector<schedule_params> schedules, computation *comp, vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes){
    deque<state*> q;
    state *current_state;
    int current_level = 0, cpt = 0;
    vector<vector<schedule*>> schedules_exhaustive;
    vector<configuration> stage_configurations = generate_configurations(schedules[current_level].schedule, schedules[current_level].factors, comp->variables, comp->type),
            passed_configurations;
    for (int i = 0; i < stage_configurations.size(); i++){
        q.insert(q.begin() + i, new state({stage_configurations[i]}, current_level));
    }
    while (!q.empty()){
        current_state = q.front();
        q.pop_front();
        if (current_state->is_extendable(schedules.size())){
            passed_configurations = current_state->schedules;
            current_level = current_state->level + 1;
            stage_configurations = generate_configurations(schedules[current_level].schedule, schedules[current_level].factors, current_state->schedules.back().out_variables, comp->type);
            for (int i = 0; i < stage_configurations.size(); i++){
                passed_configurations.push_back(stage_configurations[i]);
                q.insert(q.begin() + i, new state(passed_configurations, current_level));
                passed_configurations.pop_back();
            }
        }
        if (current_state->is_appliable(schedules.size())){
            (*generated_schedules).push_back(current_state->apply(comp));
            (*generated_variables).push_back(current_state->schedules.back().out_variables);
            (*schedule_classes).push_back(confs_to_sc(current_state->schedules));
            cpt++;
        }
    }
}

void generate_all_schedules_multiple_adj_comps(vector<schedule_params> schedules, vector<computation*> comps,
        vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes){
    deque<state*> q;
    state *current_state;
    int current_level = 0, cpt = 0;
    vector<vector<schedule*>> schedules_exhaustive;
    vector<configuration> stage_configurations = generate_configurations(schedules[current_level].schedule,
            schedules[current_level].factors, comps[0]->variables, comps[0]->type),passed_configurations;
    for (int i = 0; i < stage_configurations.size(); i++){
        q.insert(q.begin() + i, new state({stage_configurations[i]}, current_level));
    }
    while (!q.empty()){
        current_state = q.front();
        q.pop_front();
        if (current_state->is_extendable(schedules.size())){
            passed_configurations = current_state->schedules;
            current_level = current_state->level + 1;
            stage_configurations = generate_configurations(schedules[current_level].schedule, schedules[current_level].factors,
                    current_state->schedules.back().out_variables, comps[0]->type);
            for (int i = 0; i < stage_configurations.size(); i++){
                passed_configurations.push_back(stage_configurations[i]);
                q.insert(q.begin() + i, new state(passed_configurations, current_level));
                passed_configurations.pop_back();
            }
        }
        if (current_state->is_appliable(schedules.size())){
            (*generated_schedules).push_back(current_state->apply(comps));
            (*generated_variables).push_back(current_state->schedules.back().out_variables);
            (*schedule_classes).push_back(confs_to_sc(current_state->schedules));
            cpt++;
        }
    }
}

void generate_all_schedules_shared_vars(vector<schedule_params> schedules, vector<computation*> comps,
                                               vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables,
                                               vector<schedules_class*> *schedule_classes, vector<variable *> variables){
    deque<state*> q;
    state *current_state;
    int current_level = 0, cpt = 0;
    vector<vector<schedule*>> schedules_exhaustive;
    vector<configuration> stage_configurations = generate_configurations(schedules[current_level].schedule,
                                                                         schedules[current_level].factors, variables, comps[0]->type),passed_configurations;
    for (int i = 0; i < stage_configurations.size(); i++){
        q.insert(q.begin() + i, new state({stage_configurations[i]}, current_level));
    }
    while (!q.empty()){
        current_state = q.front();
        q.pop_front();
        if (current_state->is_extendable(schedules.size())){
            passed_configurations = current_state->schedules;
            current_level = current_state->level + 1;
            stage_configurations = generate_configurations(schedules[current_level].schedule, schedules[current_level].factors,
                                                           current_state->schedules.back().out_variables, comps[0]->type);
            for (int i = 0; i < stage_configurations.size(); i++){
                passed_configurations.push_back(stage_configurations[i]);
                q.insert(q.begin() + i, new state(passed_configurations, current_level));
                passed_configurations.pop_back();
            }
        }
        if (current_state->is_appliable(schedules.size())){
            (*generated_schedules).push_back(current_state->apply(comps));
            (*generated_variables).push_back(current_state->schedules.back().out_variables);
            (*schedule_classes).push_back(confs_to_sc(current_state->schedules));
            cpt++;
        }
    }
}



void generate_random_schedules(int nb_schedules, vector<schedule_params> schedules, computation *comp,
                              vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes){
    int num, n_attempts = 5, attempts = 0;
    string s;
    vector<variable*> computation_variables;
    vector<schedule*> generated_schedule;
    vector<configuration> configs;
    configuration conf;
    (*generated_schedules).push_back(generated_schedule);
    (*generated_variables).push_back(computation_variables);
    conf.schedule = NONE;
    configs.push_back(conf);
    configs.push_back(conf);
    configs.push_back(conf);
    (*schedule_classes).push_back(confs_to_sc(configs));
    for (int i = 0; i < nb_schedules; ++i) {
        generated_schedule.clear();
        computation_variables = comp->variables;
        num = rand() % ((int) pow(2.0, schedules.size()) - 1) + 1;
        s = to_base_2(num, schedules.size());
        for (int j = 0; j < schedules.size(); ++j) {
            if (s[j] == '1') {
                attempts = 0;
                conf = random_conf(schedules[j], comp->type, computation_variables);
                while ((!is_valid(conf)) && (attempts < n_attempts)){
                    conf = random_conf(schedules[j], comp->type, computation_variables);
                    attempts++;
                }
                if (attempts == n_attempts) continue;
                computation_variables = conf.out_variables;
                if (conf.schedule != NONE) {
                    generated_schedule.push_back(new schedule({comp}, conf.schedule, conf.factors, conf.in_variables));
                }
                configs.push_back(conf);
            }
        }
        (*generated_schedules).push_back(generated_schedule);
        (*generated_variables).push_back(computation_variables);
        (*schedule_classes).push_back(confs_to_sc(configs));
    }
}

void generate_random_schedules_multiple_adj_comps(int nb_schedules, vector<schedule_params> schedules, vector<computation*> comps,
                               vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes){
    int num, n_attempts = 5, attempts = 0;
    string s;
    vector<variable*> computation_variables;
    vector<schedule*> generated_schedule;
    vector<configuration> configs;
    configuration conf;
    (*generated_schedules).push_back(generated_schedule);
    (*generated_variables).push_back(computation_variables);
    conf.schedule = NONE;
    configs.push_back(conf);
    configs.push_back(conf);
    configs.push_back(conf);
    (*schedule_classes).push_back(confs_to_sc(configs));
    for (int i = 0; i < nb_schedules; ++i) {
        generated_schedule.clear();
        computation_variables = comps[0]->variables;
        num = rand() % ((int) pow(2.0, schedules.size()) - 1) + 1;
        s = to_base_2(num, schedules.size());
        for (int j = 0; j < schedules.size(); ++j) {
            if (s[j] == '1') {
                attempts = 0;
                conf = random_conf(schedules[j], comps[0]->type, computation_variables);
                while ((!is_valid(conf)) && (attempts < n_attempts)){
                    conf = random_conf(schedules[j], comps[0]->type, computation_variables);
                    attempts++;
                }
                if (attempts == n_attempts) continue;
                computation_variables = conf.out_variables;
                if (conf.schedule != NONE) {
                    generated_schedule.push_back(new schedule(comps, conf.schedule, conf.factors, conf.in_variables));
                }
                configs.push_back(conf);
            }
        }
        (*generated_schedules).push_back(generated_schedule);
        (*generated_variables).push_back(computation_variables);
        (*schedule_classes).push_back(confs_to_sc(configs));
    }
}



schedules_class *confs_to_sc(vector<configuration> schedules){
    tiling_class *tc;
    vector<int> interchange_dims;
    int unrolling_factor, pos_interchange = find_schedule(schedules, INTERCHANGE), pos_tiling_2D = find_schedule(schedules, TILE_2), pos_tiling_3D = find_schedule(schedules, TILE_3), pos_unrolling = find_schedule(schedules, UNROLL);
    (pos_interchange != -1) ?
            interchange_dims = {schedules[pos_interchange].in_variables[0]->id, schedules[pos_interchange].in_variables[1]->id} :
            interchange_dims = {};
    (pos_tiling_2D != -1) ?
            tc = new tiling_class(2, {schedules[pos_tiling_2D].in_variables[0]->id, schedules[pos_tiling_2D].in_variables[1]->id}, {schedules[pos_tiling_2D].factors[0], schedules[pos_tiling_2D].factors[1]}) :
            (pos_tiling_3D != -1) ?
                tc = new tiling_class(3, {schedules[pos_tiling_3D].in_variables[0]->id, schedules[pos_tiling_3D].in_variables[1]->id, schedules[pos_tiling_3D].in_variables[2]->id}, {schedules[pos_tiling_3D].factors[0], schedules[pos_tiling_3D].factors[1], schedules[pos_tiling_3D].factors[2]}) :
                tc = nullptr;
    (pos_unrolling != -1) ? unrolling_factor = schedules[pos_unrolling].factors[0] : unrolling_factor = (int)NULL;
    return new schedules_class(interchange_dims, unrolling_factor, tc);
}

string to_base_2(int num, int nb_pos){
    string vals = "01";
    string num_to_base_2;
    while (num > 0) {
        num_to_base_2 = vals[num % 2] + num_to_base_2;
        num /= 2;
    }
    int nb_zeros = nb_pos - num_to_base_2.size();
    for (int i = 0; i < nb_zeros; ++i) {
        num_to_base_2 = "0" + num_to_base_2;
    }
    return num_to_base_2;
}
