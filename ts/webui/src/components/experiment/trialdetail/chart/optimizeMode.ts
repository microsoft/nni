function optimizeModeValue(optimizeMode: string) {
    if (optimizeMode === 'unknown') {
        // this value is from experiment.ts file optimizeMode function
        return 'maximize';
    } else {
        return optimizeMode;
    }
}

export { optimizeModeValue };
