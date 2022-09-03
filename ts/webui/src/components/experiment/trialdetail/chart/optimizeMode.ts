function optimizeModeValue(optimizeMode: string) {
    // 'unknown' value is from experiment.ts file optimizeMode function
    return optimizeMode === 'unknown' ? 'maximize' : optimizeMode;
}

export { optimizeModeValue };
