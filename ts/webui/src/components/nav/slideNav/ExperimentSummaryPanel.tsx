import React, { useState, useEffect } from 'react';
import { Stack, PrimaryButton, DefaultButton, Panel, StackItem } from '@fluentui/react';
import MonacoEditor from 'react-monaco-editor';
import { downFile } from '@static/function';
import { DRAWEROPTION } from '@static/const';
import { EXPERIMENT, TRIALS } from '@static/datamodel';
import { caclMonacoEditorHeight } from '@static/function';
import '@style/logPanel.scss';

// TODO: the same as the file LogPanel.tsx, should clear the timerIdList rather than only the timer Id

interface ExpPanelProps {
    closeExpPanel: () => void;
    experimentProfile: object;
}

const ExperimentSummaryPanel = (props: ExpPanelProps): any => {
    const { closeExpPanel } = props;

    // experiment -> experimentSummaryData
    const [experiment, setExperiment] = useState('' as string);
    const [expPanelHeight, setExpPanelHeight] = useState(window.innerHeight as number);
    let refreshId: number = 0; // TODO: use list rather than number

    const getExperimentContent = (): void => {
        const experimentData = JSON.parse(JSON.stringify(props.experimentProfile));
        const trialMessagesArr = TRIALS.getTrialJobList();
        const interResultList = TRIALS.getMetricsList();
        Object.keys(trialMessagesArr).map(item => {
            // not deal with trial's hyperParameters
            const trialId = trialMessagesArr[item].trialJobId;
            // add intermediate result message
            trialMessagesArr[item].intermediate = [];
            Object.keys(interResultList).map(key => {
                const interId = interResultList[key].trialJobId;
                if (trialId === interId) {
                    trialMessagesArr[item].intermediate.push(interResultList[key]);
                }
            });
        });
        const result = {
            experimentParameters: experimentData,
            trialMessage: trialMessagesArr
        };
        setExperiment(JSON.stringify(result, null, 4));

        if (['DONE', 'ERROR', 'STOPPED', 'VIEWED'].includes(EXPERIMENT.status)) {
            if (refreshId !== undefined) {
                window.clearInterval(refreshId);
            }
        }
    };

    const downExperimentParameters = (): void => {
        downFile(experiment, 'experiment.json');
    };

    const onWindowResize = (): void => {
        setExpPanelHeight(window.innerHeight);
    };

    useEffect(() => {
        getExperimentContent();
        refreshId = window.setInterval(getExperimentContent, 10000);
        window.addEventListener('resize', onWindowResize);
        return function () {
            window.clearTimeout(refreshId);
            window.removeEventListener('resize', onWindowResize);
        };
    }, []); // DidMount and willUnMount component
    const monacoEditorHeight = caclMonacoEditorHeight(expPanelHeight);

    return (
        <Panel isOpen={true} hasCloseButton={false} isLightDismiss={true} onLightDismissClick={closeExpPanel}>
            <div className='panel'>
                <div className='panelName'>Summary</div>
                <MonacoEditor
                    width='100%'
                    height={monacoEditorHeight}
                    language='json'
                    value={experiment}
                    options={DRAWEROPTION}
                />
                <Stack horizontal className='buttons'>
                    <StackItem grow={50} className='download'>
                        <PrimaryButton text='Download' onClick={downExperimentParameters} />
                    </StackItem>
                    <StackItem grow={50} className='close'>
                        <DefaultButton text='Close' onClick={closeExpPanel} />
                    </StackItem>
                </Stack>
            </div>
        </Panel>
    );
};

export default ExperimentSummaryPanel;
