import * as React from 'react';
import { Experiment } from '../../static/interface';
import MonacoEditor from 'react-monaco-editor';
import { MONACO } from '../../static/const';

interface TrialInfoProps {
    tiralProInfo: Experiment;
}

class TrialInfo extends React.Component<TrialInfoProps, {}> {

    constructor(props: TrialInfoProps) {
        super(props);
    }

    render() {
        const { tiralProInfo } = this.props;
        const showProInfo = [];
        showProInfo.push({
            revision: tiralProInfo.revision,
            authorName: tiralProInfo.author,
            trialConcurrency: tiralProInfo.runConcurren,
            tuner: tiralProInfo.tuner,
            assessor: tiralProInfo.assessor ? tiralProInfo.assessor : undefined,
            advisor: tiralProInfo.advisor ? tiralProInfo.advisor : undefined,
            clusterMetaData: tiralProInfo.clusterMetaData ? tiralProInfo.clusterMetaData : undefined
        });
        return (
            <div className="profile">
                <MonacoEditor
                    width="100%"
                    height="380"
                    language="json"
                    theme="vs-light"
                    value={JSON.stringify(showProInfo[0], null, 2)}
                    options={MONACO}
                />
            </div>
        );
    }
}

export default TrialInfo;