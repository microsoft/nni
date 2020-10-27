import * as React from 'react';
import { TRIALS } from '../../static/datamodel';
import { formatAccuracy } from '../../static/function';

interface DefaultMetricProps {
    trialId: string;
}

class DefaultMetric extends React.Component<DefaultMetricProps, {}> {
    constructor(props: DefaultMetricProps) {
        super(props);
    }

    render(): React.ReactNode {
        const accuracy = TRIALS.getTrial(this.props.trialId).accuracy;
        return <div className='succeed-padding'>{accuracy !== undefined ? formatAccuracy(accuracy) : '--'}</div>;
    }
}

export default DefaultMetric;
