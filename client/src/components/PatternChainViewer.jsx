import React from 'react';
import { ArrowRight, Brain } from 'lucide-react';

function PatternChainViewer({ selectedPattern, chainData, onClose }) {
    if (!selectedPattern || !chainData) return null;

    const data = chainData.find(item => item.pattern === selectedPattern);
    if (!data) return null;

    // Build a chain by recursively following the highest confidence patterns
    const buildPredecessorChain = (pattern, maxLength = 3, visited = new Set()) => {
        if (maxLength === 0 || visited.has(pattern)) return [];
        
        const patternData = chainData.find(p => p.pattern === pattern);
        if (!patternData || patternData.predecessors.length === 0) return [];
        
        visited.add(pattern);
        const topPred = patternData.predecessors[0];
        const prevChain = buildPredecessorChain(topPred.pattern, maxLength - 1, visited);
        
        return [...prevChain, { pattern: topPred.pattern, confidence: topPred.count }];
    };

    const buildSuccessorChain = (pattern, maxLength = 3, visited = new Set()) => {
        if (maxLength === 0 || visited.has(pattern)) return [];
        
        const patternData = chainData.find(p => p.pattern === pattern);
        if (!patternData || patternData.successors.length === 0) return [];
        
        visited.add(pattern);
        const topSucc = patternData.successors[0];
        const nextChain = buildSuccessorChain(topSucc.pattern, maxLength - 1, visited);
        
        return [{ pattern: topSucc.pattern, confidence: topSucc.count }, ...nextChain];
    };

    const predecessorChain = buildPredecessorChain(selectedPattern);
    const successorChain = buildSuccessorChain(selectedPattern);

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl shadow-2xl max-w-7xl w-full max-h-[90vh] overflow-hidden flex flex-col">
                
                {/* Header */}
                <div className="bg-gradient-to-r from-[#006747] to-[#008556] text-white p-6 flex justify-between items-center">
                    <div>
                        <div className="flex items-center gap-2 mb-2">
                            <Brain className="w-6 h-6" />
                            <h2 className="text-2xl font-bold">Pattern Chain Timeline</h2>
                        </div>
                        <p className="text-green-100 text-sm">Most likely behavioral sequence for {selectedPattern}</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-white hover:bg-white hover:bg-opacity-20 rounded-lg p-2 transition-all"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Main Content */}
                <div className="flex-1 overflow-y-auto p-8">
                    
                    {/* Timeline Header */}
                    <div className="text-center mb-8">
                        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-2">Predicted Pattern Sequence</h3>
                        <p className="text-xs text-gray-400">Full behavioral chain based on ML predictions</p>
                    </div>

                    {/* Main Timeline Chain */}
                    <div className="mb-12">
                        <div className="flex items-center justify-center gap-2 flex-wrap overflow-x-auto pb-4">
                            
                            {/* BEFORE: Predecessor Chain */}
                            {predecessorChain.length > 0 ? (
                                predecessorChain.map((item, index) => (
                                    <React.Fragment key={`pred-${index}`}>
                                        <div className="flex flex-col items-center">
                                            <div className="bg-[#CDB87D] text-white px-4 py-2 rounded-lg font-bold shadow-md min-w-[120px] text-center text-sm">
                                                {item.pattern}
                                            </div>
                                            <span className="text-xs text-gray-400 mt-1">{item.confidence}%</span>
                                        </div>
                                        <ArrowRight className="w-6 h-6 text-gray-400 flex-shrink-0" />
                                    </React.Fragment>
                                ))
                            ) : (
                                <>
                                    <div className="flex flex-col items-center">
                                        <div className="bg-gray-200 text-gray-400 px-4 py-2 rounded-lg font-bold min-w-[80px] text-center text-sm border-2 border-dashed border-gray-300">
                                            ...
                                        </div>
                                    </div>
                                    <ArrowRight className="w-6 h-6 text-gray-300 flex-shrink-0" />
                                </>
                            )}

                            {/* CURRENT: Selected Pattern (Center) */}
                            <div className="flex flex-col items-center">
                                <div className="bg-[#006747] text-white px-6 py-3 rounded-xl font-bold shadow-lg ring-4 ring-[#CDB87D] ring-opacity-40 min-w-[140px] text-center">
                                    {selectedPattern}
                                </div>
                                <span className="text-xs text-[#006747] mt-1 font-bold uppercase">Selected</span>
                            </div>

                            {/* AFTER: Successor Chain */}
                            {successorChain.length > 0 ? (
                                successorChain.map((item, index) => (
                                    <React.Fragment key={`succ-${index}`}>
                                        <ArrowRight className="w-6 h-6 text-gray-400 flex-shrink-0" />
                                        <div className="flex flex-col items-center">
                                            <div className="bg-[#CDB87D] text-white px-4 py-2 rounded-lg font-bold shadow-md min-w-[120px] text-center text-sm">
                                                {item.pattern}
                                            </div>
                                            <span className="text-xs text-gray-400 mt-1">{item.confidence}%</span>
                                        </div>
                                    </React.Fragment>
                                ))
                            ) : (
                                <>
                                    <ArrowRight className="w-6 h-6 text-gray-300 flex-shrink-0" />
                                    <div className="flex flex-col items-center">
                                        <div className="bg-gray-200 text-gray-400 px-4 py-2 rounded-lg font-bold min-w-[80px] text-center text-sm border-2 border-dashed border-gray-300">
                                            ...
                                        </div>
                                    </div>
                                </>
                            )}

                        </div>

                        {/* Timeline Direction Labels */}
                        <div className="flex justify-between mt-6 px-4">
                            <div className="text-left">
                                <p className="text-xs font-bold text-gray-600 uppercase">← Earlier Patterns</p>
                            </div>
                            <div className="text-center">
                                <p className="text-xs font-bold text-[#006747] uppercase">Current</p>
                            </div>
                            <div className="text-right">
                                <p className="text-xs font-bold text-gray-600 uppercase">Later Patterns →</p>
                            </div>
                        </div>
                    </div>

                    {/* Alternative Patterns */}
                    <div className="border-t border-gray-200 pt-6">
                        <h4 className="text-sm font-bold text-gray-700 mb-4 text-center">Alternative Pattern Transitions</h4>
                        
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            
                            {/* Other Predecessors */}
                            <div>
                                <h5 className="font-semibold text-gray-600 mb-3 text-xs uppercase tracking-wide flex items-center gap-2">
                                    <span className="bg-gray-100 px-2 py-1 rounded">Before {selectedPattern}</span>
                                </h5>
                                {data.predecessors.length > 0 ? (
                                    <div className="space-y-2">
                                        {data.predecessors.slice(0, 5).map((pred, index) => (
                                            <div key={index} className="flex items-center justify-between bg-gray-50 p-2 rounded border border-gray-200">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-xs text-gray-400 font-bold">#{index + 1}</span>
                                                    <span className="text-sm font-medium text-gray-700">{pred.pattern}</span>
                                                </div>
                                                <span className="text-xs font-bold text-gray-600 bg-white px-2 py-1 rounded">{pred.count}%</span>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-xs text-gray-400 italic text-center py-4">No patterns detected</p>
                                )}
                            </div>

                            {/* Other Successors */}
                            <div>
                                <h5 className="font-semibold text-gray-600 mb-3 text-xs uppercase tracking-wide flex items-center gap-2">
                                    <span className="bg-gray-100 px-2 py-1 rounded">After {selectedPattern}</span>
                                </h5>
                                {data.successors.length > 0 ? (
                                    <div className="space-y-2">
                                        {data.successors.slice(0, 5).map((succ, index) => (
                                            <div key={index} className="flex items-center justify-between bg-gray-50 p-2 rounded border border-gray-200">
                                                <div className="flex items-center gap-2">
                                                    <span className="text-xs text-gray-400 font-bold">#{index + 1}</span>
                                                    <span className="text-sm font-medium text-gray-700">{succ.pattern}</span>
                                                </div>
                                                <span className="text-xs font-bold text-gray-600 bg-white px-2 py-1 rounded">{succ.count}%</span>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-xs text-gray-400 italic text-center py-4">No patterns detected</p>
                                )}
                            </div>

                        </div>
                    </div>

                    {/* Info Box */}
                    <div className="mt-6 bg-[#fdfcf6] border border-[#CDB87D] border-opacity-30 rounded-lg p-4">
                        <div className="flex items-start gap-3">
                            <div className="bg-[#CDB87D] bg-opacity-20 rounded-full p-2 flex-shrink-0">
                                <Brain className="w-5 h-5 text-[#006747]" />
                            </div>
                            <div className="flex-1">
                                <h4 className="font-semibold text-gray-800 text-sm mb-1">Understanding the Chain</h4>
                                <p className="text-xs text-gray-600 leading-relaxed">
                                    This shows the most likely <strong>sequence</strong> of patterns. The chain is built by following the highest-confidence 
                                    transitions from the ML model. Patterns can repeat in real user behavior, and the model captures these repetitions. 
                                    Percentages show the confidence for each transition.
                                </p>
                            </div>
                        </div>
                    </div>

                </div>

                {/* Footer */}
                <div className="bg-gray-50 px-6 py-4 flex justify-end border-t border-gray-200">
                    <button
                        onClick={onClose}
                        className="bg-[#006747] hover:bg-[#005538] text-white px-6 py-2 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg"
                    >
                        Close
                    </button>
                </div>

            </div>
        </div>
    );
}

export default PatternChainViewer;
