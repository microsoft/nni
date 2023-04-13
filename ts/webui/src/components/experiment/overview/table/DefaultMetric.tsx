import * as React from 'react';
import { TRIALS } from '@static/datamodel';
import { formatAccuracy } from '@static/function';

// oview page table: default metric column render
interface DefaultMetricProps {
    trialId: string;
}

const DefaultMetric = (props: DefaultMetricProps): any => {
    const accuracy = TRIALS.getTrial(props.trialId).accuracy;
    return <div className='succeed-padding metric'>{accuracy !== undefined ? formatAccuracy(accuracy) : '--'}</div>;
};

export default DefaultMetric;
