import * as React from 'react';
import MonacoEditor from 'react-monaco-editor';
import { MONACO } from '../../static/const';

interface TrialInfoProps {
    experiment: object;
}

class TrialInfo extends React.Component<TrialInfoProps, {}> {

    constructor(props: TrialInfoProps) {
        super(props);
    }

    componentWillReceiveProps(nextProps: TrialInfoProps) {
        const experiments = nextProps.experiment;
        Object.keys(experiments).map(key => {
            switch (key) {
                case 'id':
                case 'logDir':
                case 'startTime':
                case 'endTime':
                    experiments[key] = undefined;
                    break;
                case 'params':
                    const params = experiments[key];
                    Object.keys(params).map(item => {
                        if (item === 'experimentName' || item === 'searchSpace'
                            || item === 'trainingServicePlatform') {
                            params[item] = undefined;
                        }
                    });
                    break;
                default:
            }
        });
    }

    render() {
        const { experiment } = this.props;
        return (
            <div className="profile">
                <MonacoEditor
                    width="100%"
                    height="361"
                    language="json"
                    theme="vs-light"
                    value={JSON.stringify(experiment, null, 2)}
                    options={MONACO}
                />
            </div>
        );
    }
}

export default TrialInfo;
