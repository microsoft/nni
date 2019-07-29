import * as React from 'react';
import { TableObj } from '../../static/interface';

interface IntermediateValProps {
    record: TableObj;
}

class IntermediateVal extends React.Component<IntermediateValProps, {}> {

    constructor(props: IntermediateValProps) {
        super(props);

    }

    render() {
        const { record } = this.props;
        const interArr = record.description.intermediate;
        let lastVal;
        let wei = 0;
        if (interArr !== undefined) {
            lastVal = interArr[interArr.length - 1];
        }
        let result: string = JSON.stringify(lastVal);
        if (lastVal !== undefined) {
            if (lastVal.toString().indexOf('.') !== -1) {
                wei = lastVal.toString().length - lastVal.toString().indexOf('.') - 1;
                if (wei > 6) {
                    result = `${lastVal.toFixed(6)}`;
                }
            }
            // some trial haven't final result
            if (record.acc !== undefined) {
                if (record.acc.default !== undefined) {
                    result = `${result} (FINAL)`;
                }
            } else {
                result = `${result} (LATEST)`;
            }
        } else {
            result = '--';
        }
        return (
            <div>{result}</div>
        );
    }
}

export default IntermediateVal;
