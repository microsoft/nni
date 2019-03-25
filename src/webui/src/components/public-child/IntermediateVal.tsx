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
        if (interArr !== undefined) {
            lastVal = interArr[interArr.length - 1];
        }
        let wei = 0;
        if (lastVal !== undefined) {
            if (lastVal.toString().indexOf('.') !== -1) {
                wei = lastVal.toString().length - lastVal.toString().indexOf('.') - 1;
            }
        }
        return (
            <div>
                {
                    lastVal !== undefined
                        ?
                        wei > 6
                            ?
                            lastVal.toFixed(6)
                            :
                            lastVal
                        :
                        '--'
                }
            </div>
        );
    }
}

export default IntermediateVal;