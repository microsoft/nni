import * as React from 'react';
import { TableObj } from '../../static/interface';

interface DefaultMetricProps {
    record: TableObj;
}

class DefaultMetric extends React.Component<DefaultMetricProps, {}> {

    constructor(props: DefaultMetricProps) {
        super(props);

    }

    render() {
        const { record } = this.props;
        let accuracy;
        if (record.acc !== undefined) {
            accuracy = record.acc.default;
        }
        let wei = 0;
        if (accuracy) {
            if (accuracy.toString().indexOf('.') !== -1) {
                wei = accuracy.toString().length - accuracy.toString().indexOf('.') - 1;
            }
        }
        return (
            <div>
                {
                    record.acc && record.acc.default
                        ?
                        wei > 6
                            ?
                            JSON.parse(record.acc.default).toFixed(6)
                            :
                            record.acc.default
                        :
                        '--'
                }
            </div>
        );
    }
}

export default DefaultMetric;