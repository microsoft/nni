import * as React from 'react';
import { TRIALS } from '../../static/datamodel';

interface IntermediateValProps {
    trialId: string;
}

class IntermediateVal extends React.Component<IntermediateValProps, {}> {
    constructor(props: IntermediateValProps) {
        super(props);
    }

    render(): React.ReactNode {
        return <div>{TRIALS.getTrial(this.props.trialId).formatLatestAccuracy()}</div>;
    }
}

export default IntermediateVal;
