
#include <iostream>
#include "bridging_sampler.h"


void print_state(const bridgesamp::BridgingSampler& sampler) {
    std::cout << std::endl;
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

int main() {
    bridgesamp::BridgingSampler sampler(
        2, 2,
        [](const std::vector<uint16_t>& s) -> double {
            if(s[0] == s[1]) {
                return -100.;
            } else if(s[0] == 0) {
                return 0.;
            } else {
                return -1.;
            }
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

    std::vector<std::vector<uint16_t>> chain;

    print_state(sampler);
    std::cout << std::endl;
    chain.push_back(sampler.get_state());

    for(int n=0; n<1000; n++) {
        sampler.step();
        std::cout << std::endl;
        print_state(sampler);
        std::cout << std::endl;
        chain.push_back(sampler.get_state());
    }
    
    std::map<std::vector<uint16_t>, uint32_t> n_visits;

    for(int n=0; n<100000; n++) {
        sampler.step();
        std::cout << std::endl;
        print_state(sampler);
        std::cout << std::endl;
        chain.push_back(sampler.get_state());
        n_visits[sampler.get_state()]++;
    }

    std::cout << std::endl
              << "chain:"
              << std::endl;
    for(auto& c : chain) {
        for(auto s : c) {
            std::cout << s << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << std::endl
              << "# of visits:"
              << std::endl;
    for(auto& v : n_visits) {
        for(auto s : v.first) {
            std::cout << s << " ";
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

    return 0;
}
