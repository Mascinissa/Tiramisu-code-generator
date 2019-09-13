#include <random>
#include <algorithm>
#include "tiramisu_code_generator.h"

//=====================================================================tiramisu_code_generator==========================================================================================================


bool inf(int i, int j) { return (i < j); }



vector<variable *> generate_variables(int nb_variables, int from, int *inf_values, vector<constant *> constants) {
    vector<variable *> variables;
    for (int i = from; i < nb_variables + from; ++i) {
        variables.push_back(new variable("i" + to_string(i), i, inf_values[i - from], constants[i - from]));
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
    offset = rand() % offset + 1;
    int id = 0, nb, nb_dims = (rand() % (max_nb_dims - 1)) + 2, sum = 0, const_sum = MAX_MEMORY_SIZE - nb_dims * MIN_LOOP_DIM;
    vector<int> computation_dims;
    for (int i = 0; i < nb_dims; ++i) {
        computation_dims.push_back((rand() % MAX_CONST_VALUE) + 1);
        sum += computation_dims[i];
    }

    for (int i = 0; i < nb_dims; ++i) {
        computation_dims[i] *= const_sum;
        computation_dims[i] /= sum;
        computation_dims[i] += MIN_LOOP_DIM;
        computation_dims[i] = (int) pow(2.0, computation_dims[i]);
    }
    (*stats)[nb_dims - 2]->data_sizes[const_sum]++;


    (*stats)[nb_dims - 2]->nb_progs++;

    string function_name = "function" + to_string(code_id);
    vector<variable *> variables, variables_stencils;
    vector<variable*> all_vars;
    vector<computation *> computations;
    vector<buffer *> buffers;
    vector<input *> inputs;
    vector<computation_abstract *> abs, abs1;

    int *variables_min_values = new int[computation_dims.size()];
    vector<constant *> variable_max_values;
    int *variables_min_values_stencils = new int[computation_dims.size()];
    vector<constant *> variable_max_values_stencils;

    nb = rand() % nb_stages + 1;
    //nb = nb_stages;
    bool st = false;

    int inp, random_index;

    map<int, vector<int>> indexes;
    vector<vector<variable *>> variables_inputs;
    vector<variable *> vars_inputs;

    vector<int> types = computation_types(nb_stages, probs), new_indices;
    computation *stage_computation;

    for (int i = 0; i < computation_dims.size(); ++i) {
//        variables_min_values[i] = 0;
//        variable_max_values.push_back(new constant("c" + to_string(i), computation_dims[i]));
//        variables_min_values_stencils[i] = offset;
//        variable_max_values_stencils.push_back(
//                new constant("c" + to_string(i) + " - " + to_string(offset), computation_dims[i] - offset));
        variables_min_values[i] = 0;
        variable_max_values.push_back(new constant("c" + to_string(i), computation_dims[i]));
        variables_min_values_stencils[i] = 0;
        variable_max_values_stencils.push_back(
                new constant("c" + to_string(i) + " - " + to_string(offset)+ " - " + to_string(offset), computation_dims[i] - offset - offset));
    }

    variables=generate_variables(computation_dims.size(), 0, variables_min_values, variable_max_values);
    variables_stencils=generate_variables(computation_dims.size(),  100,
                                                        variables_min_values_stencils, variable_max_values_stencils);


    // since all computations must have same iterators, if one comp is a stencil all the other comps will have stencil iterators
    vector <variable*> stencils_input_variables=variables;
    if(find(types.begin(), types.end(), STENCIL) != types.end()) {

        variables = variables_stencils;

    }


    for (int i = 0; i < nb; ++i) {

        inp = rand() % nb_inputs + 1;
        switch (types[i]) {
            case ASSIGNMENT:
                stage_computation = generate_computation("comp" + to_string(i), variables, ASSIGNMENT, {}, {}, 0,
                                                         *default_type_tiramisu, id++);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), computation_dims, OUTPUT_BUFFER, &abs));
                break;
            case ASSIGNMENT_INPUTS:
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
                vector<int> var_nums_copy;
                vector<int> var_nums_copy_copy;

                for (int l = 0; l < variables_stencils.size(); ++l) {
                    var_nums_copy_copy.push_back(i);
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
                //clearin abs for not using the previous computation as input
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
                stage_computation = generate_computation("comp" + to_string(i), variables_stencils, STENCIL, abs,
                                                         var_nums_copy, offset, *default_type_tiramisu, id++);
                computations.push_back(stage_computation);
                abs = {stage_computation};
                buffers.push_back(new buffer("buf" + to_string(i), computation_dims, OUTPUT_BUFFER, &abs));
                break;
        }
        (*stats)[computation_dims.size() - 2]->types[types[i]]->nb_assignments++;
    }

   /* for (int k = 0; (k < variables_stencils.size()) && st; ++k) {
        variables.push_back(variables_stencils[k]);
    }*/

    for (int k = 0; (k < stencils_input_variables.size()) && st; ++k) {
        variables.push_back(stencils_input_variables[k]);
    }

    for (int j = 0; j < variables.size(); ++j) {
            all_vars.push_back(variables[j]);
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
        generate_all_schedules_multiple_adj_comps(schedules_parameters, computations, &schedules_exhaustive, &variables_exhaustive,
                                &schedule_classes);
    }
    else {
       // generate_random_schedules(nb_rand_schedules, schedules_parameters, computations[0], &schedules_exhaustive,
          //                        &variables_exhaustive, &schedule_classes);
        generate_random_schedules_multiple_adj_comps(nb_rand_schedules, schedules_parameters, computations, &schedules_exhaustive,
                                                     &variables_exhaustive, &schedule_classes);
    }


    tiramisu_code *code;

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

void generate_all_schedules_multiple_adj_comps(vector<schedule_params> schedules, vector<computation*> comps, vector<vector <schedule*>> *generated_schedules, vector<vector<variable*>> *generated_variables, vector<schedules_class*> *schedule_classes){
    deque<state*> q;
    state *current_state;
    int current_level = 0, cpt = 0;
    vector<vector<schedule*>> schedules_exhaustive;
    vector<configuration> stage_configurations = generate_configurations(schedules[current_level].schedule, schedules[current_level].factors, comps[0]->variables, comps[0]->type),
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
            stage_configurations = generate_configurations(schedules[current_level].schedule, schedules[current_level].factors, current_state->schedules.back().out_variables, comps[0]->type);
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
