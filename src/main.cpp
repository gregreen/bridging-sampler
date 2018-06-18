
#include <iostream>
#include "bridging_sampler.h"

void print_state(const bridgesamp::BridgingSampler& sampler) {
    std::cout << "* ";

    for(auto s : sampler.get_state()) {
        if(s == sampler.get_n_samples()) {
            std::cout << "- ";
        } else {
            std::cout << s << " ";
        }
    }

    std::cout << "(" << sampler.get_state_rank() << ")"
              << std::endl << std::endl;

    for(auto it=sampler.cbegin(); it!=sampler.cend(); ++it) {
        for(auto s : it->first) {
            if(s == sampler.get_n_samples()) {
                std::cout << "- ";
            } else {
                std::cout << s << " ";
            }
        }
        std::cout << ": " << std::exp(it->second.logp) << std::endl;
    }
}

void binary_labeling_toy() {
    // Set up the sampler, providing a function that calculates
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
}


void toy() {
    bridgesamp::BridgingSampler sampler(
        3, 5,
        [](const std::vector<uint16_t>& s) -> double {
            for(auto x : s) {
                if(x != s[0]) {
                    return -100.;
                }
            }
            return 0.;
            //if(s[0] == s[1]) {
            //    return -100.;
            //} else {
            //    return 0.;
            //}
        }
    );
    
    sampler.randomize_state();

    //for(int n=0; n<5; n++) {
    //    for(int i=0; i<3; i++) {
    //        std::cout << std::endl;
    //        sampler.lazy_gibbs(0);
    //        print_state(sampler);

    //        std::cout << std::endl;
    //        sampler.transition_backward();
    //        print_state(sampler);
    //    }
    //    
    //    for(int i=0; i<3; i++) {
    //        std::cout << std::endl;
    //        sampler.lazy_gibbs(0);
    //        print_state(sampler);

    //        std::cout << std::endl;
    //        sampler.transition_forward();
    //        print_state(sampler);
    //    }
    //}

    std::cout << "Initial sampler state:" << std::endl;
    print_state(sampler);
    std::cout << std::endl;
    //chain.push_back(sampler.get_state());

    for(int n=0; n<1000; n++) {
        sampler.step();
        //std::cout << std::endl;
        //print_state(sampler);
        //std::cout << std::endl;
        //chain.push_back(sampler.get_state());
    }
    
    //std::vector<std::vector<uint16_t>> chain;
    std::map<std::vector<uint16_t>, uint32_t> n_visits;

    for(int n=0; n<100000; n++) {
        sampler.step();
        //std::cout << std::endl;
        //print_state(sampler);
        //std::cout << std::endl;
        //chain.push_back(sampler.get_state());
        n_visits[sampler.get_state()]++;
    }

    //std::cout << std::endl
    //          << "chain:"
    //          << std::endl;
    //for(auto& c : chain) {
    //    for(auto s : c) {
    //        std::cout << s << " ";
    //    }
    //    std::cout << std::endl;
    //}
    
    std::cout << "Final sampler state:" << std::endl;
    print_state(sampler);
    std::cout << std::endl;
    
    std::cout << std::endl
              << "# of visits:"
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

    //unsigned int n = 4;
    //unsigned int r = 2;
    //std::vector<unsigned int> idx(r);
    //bridgesamp::CombinationGenerator cgen(n, r);
    //while(cgen.next(idx)) {
    //    for(auto i : idx) {
    //        std::cout << i << " ";
    //    }
    //    std::cout << std::endl;
    //}

    //std::cout << "Reset." << std::endl;
    //cgen.reset();
    //while(cgen.next(idx)) {
    //    for(auto i : idx) {
    //        std::cout << i << " ";
    //    }
    //    std::cout << std::endl;
    //}
}


int main() {
    binary_labeling_toy();

    return 0;
}
