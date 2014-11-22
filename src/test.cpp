#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <map>

using namespace std;

// Data types to represent finite state machines. A finiteStateMachine maps the string
// name of a state to the transition table for that state. The transition table is 
// represented by stateTransitions that maps each input symbol to the resulting state.
typedef std::map<std::string, std::string> stateTransitions;
typedef std::map<std::string, stateTransitions> finiteStateMachine;

string evaluate_fsm(finiteStateMachine fsm, string input, string startState);

int main (int argc, char *argv[]) {
	finiteStateMachine test;
	stateTransitions transitions;

	// State A
	transitions["0"] = "A";
	transitions["1"] = "B";
	test["A"] = transitions;
	transitions.clear();

	// State B
	transitions["0"] = "A";
	transitions["1"] = "C";
	test["B"] = transitions;
	transitions.clear();

	// State C
	transitions["0"] = "D";
	transitions["1"] = "A";
	test["C"] = transitions;
	transitions.clear();

	// State D
	transitions["0"] = "A";
	transitions["1"] = "D";
	test["D"] = transitions;
	transitions.clear();

	string input = "000110111111";

	string endState = evaluate_fsm(test, input, "A");

	cout << endState;

	return 0;
}

// Accepts a finite state machine, an input string, and a start state, and returns the 
// state that the machine will be in after executing the entire input string.
string evaluate_fsm(finiteStateMachine fsm, string input, string startState) {
	string currentState = startState;

	// Loop over each input symbol
	for (string::iterator it = input.begin(); it != input.end(); ++it) {
		string inputSymbol(1, *it);
		currentState = fsm[currentState][inputSymbol];
	}
	
	return currentState;
}