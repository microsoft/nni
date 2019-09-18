import * as React from 'react';
import { TRIALS } from '../../static/datamodel';
import { Trial } from '../../static/model/trial';
import { metricAccuracy } from '../../static/function';

interface IntermediateValProps {
    trialId: string;
}

class IntermediateVal extends React.Component<IntermediateValProps, {}> {
    constructor(props: IntermediateValProps) {
        super(props);
    }

    render() {
        const trial = TRIALS.getTrial(this.props.trialId);
        return (
            <div>{formatLatestAccuracy(trial)}</div>
        );
    }
}

function formatLatestAccuracy(trial: Trial) {
    if (trial.accuracy !== undefined) {
        return `${formatAccuracy(trial.accuracy)} (FINAL)`;
    } else if (trial.intermediateMetrics.length === 0) {
        return '--';
    } else {
        const latest = trial.intermediateMetrics[trial.intermediateMetrics.length - 1];
        return `${formatAccuracy(metricAccuracy(latest))} (LATEST)`;
    }
}

function formatAccuracy(accuracy: number) {
    // TODO: NaN
    return accuracy.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
}

export default IntermediateVal;
