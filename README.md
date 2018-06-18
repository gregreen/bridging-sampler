Bridging Sampler
================

Sampling from combinatorial spaces can be a challenging problem, particularly
when there are many constraints on the allowed states.

As an example, take the simple problem of trying to sample from a space
consisting of two bits. Assign the following probabilities to the four possible
states:

* `(0, 0)` -> 0
* `(0, 1)` -> 1
* `(1, 0)` -> 1
* `(0, 0)` -> 0

For such a small state space, we can obviously calculate the probabilities
of all possible states directly, but for larger state spaces, we have to
sample. A common means of sampling is to begin at a random point in state
space, and to then take Gibbs steps in each dimension. In our toy problem
above, that approach would not work, however. Imagine starting at `(1, 0)`.
Gibbs steps in either dimension will never successfully transition to
another state, because both `(0, 0)` and `(1, 1)` have zero probability.
The Markov Chain will never transition to the other possible solution,
`(0, 1)`, because there is no path (using Gibbs steps) between the two
solutions.

[Lin & Fisher (2012)](http://proceedings.mlr.press/v22/lin12.html)
(*Efficient Sampling from Combinatorial Space via Bridging*) presents a
method for sampling from problems like the one above, where naive
Gibbs sampling fails. Their technique, which I'll refer to as the
*bridging sampler*, samples from a larger combinatorial space, consisting
of the target space, as well as so-called *bridging states*, which
provide a path between separated solutions. These bridging states are
constructed by leaving some of the dimensions unset. In our toy problem
above, the bridging states would be

* `(-, -)`
* `(0, -)`
* `(1, -)`
* `(-, 0)`
* `(-, 1)`

where `-` indicates that a given dimension is unset. For any bridging
state, we can define *child states* by filling in the unset dimensions.
For example, the bridging state `(0, -)` has two children: `(0, 0)` and
`(0, 1)`. The bridging state `(-, -)` has four children (the remaining
bridging states). The weight assigned to a bridging state is equal to the
weights of its children.

Lin & Fisher define two new step types, *forward* and *backward transitions*,
which allow the sampler to move between the target and bridging states.
With these step types, as well as Gibbs sampling to move between target states,
the subset of states in the chain that belong to the target space are
distributed according to their weights.


Compilation
===========

In the base directory, run

    cmake .
    make


Using the Code
==============

The file `src/main.cpp` contains an example problem.

To sample from our toy model, all you need to do is to include
`"bridging_sampler.h"`, and then to write the following code:

    // Set up the sampler, providing a function that calcualtes
    // log(probability) for each state
    bridgesamp::BridgingSampler sampler(
        2, 2, // # of dimensions, # of samples per dimension
        [](const std::vector<uint16_t>& s) -> double {
            if(s[0] == s[1]) {
                return -100.;  // Zero probability for (0,0) and (1,1)
            } else {
                return 0.;     // Probability=1 for (1,0) and (0,1)
            }
        }
    );
    
    sampler.randomize_state();
    
    // Burn-in
    for(int n=0; n<1000; n++) {
        sampler.step();
    }
    
    // Store the number of visits to each state
    std::map<std::vector<uint16_t>, uint32_t> n_visits;
    
    // Sample
    for(int n=0; n<1000; n++) {
        sampler.step();
        n_visits[sampler.get_state()]++;
    }

    // Print out the number of visits to each state
    std::cout << "# of visits:"
              << std::endl;
    
    for(auto& v : n_visits) {
        for(auto s : v.first) {
            if(s == sampler.get_n_samples()) {
                std::cout << "- ";
            } else {
                std::cout << s << " ";
            }
        }
        std::cout << ": " << v.second << std::endl;
    }

This code will sample from the augmented space (including the target states
and the bridging states), and then print the number of times each state was
visited. The unallowed states, `(0, 0)` and `(1, 1)`, should remain unvisited
(after burn-in), while the states `(0, 1)` and `(1, 0)` should be visited
an approximately equal number of times.


License
=======

This code is made available under the MIT License.
