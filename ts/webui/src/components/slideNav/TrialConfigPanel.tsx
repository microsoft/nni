import * as React from 'react';
import { Stack, Panel, PrimaryButton } from '@fluentui/react';
import { EXPERIMENT } from '../../static/datamodel';
import MonacoEditor from 'react-monaco-editor';
import { MONACO } from '../../static/const';
import { AppContext } from '../../App';
import { convertDuration, convertTimeAsUnit, caclMonacoEditorHeight } from '../../static/function';
import { prettyStringify } from '../../static/json_util';
import lodash from 'lodash';
import '../../static/style/logDrawer.scss';

interface LogDrawerProps {
    hideConfigPanel: () => void;
    panelName: string;
}

interface LogDrawerState {
    panelInnerHeight: number;
    innerWidth: number;
}

/**
 * search space
 * config
 * model
 */

class TrialConfigPanel extends React.Component<LogDrawerProps, LogDrawerState> {
    constructor(props: LogDrawerProps) {
        super(props);

        this.state = {
            panelInnerHeight: window.innerHeight,
            innerWidth: window.innerWidth
        };
    }

    // use arrow function for change window size met error: this.setState is not a function
    setLogDrawerHeight = (): void => {
        this.setState(() => ({ panelInnerHeight: window.innerHeight, innerWidth: window.innerWidth }));
    };

    async componentDidMount(): Promise<void> {
        window.addEventListener('resize', this.setLogDrawerHeight);
    }

    componentWillUnmount(): void {
        window.removeEventListener('resize', this.setLogDrawerHeight);
    }

    render(): React.ReactNode {
        const { hideConfigPanel, panelName } = this.props;
        const { panelInnerHeight, innerWidth } = this.state;
        const monacoEditorHeight = caclMonacoEditorHeight(panelInnerHeight);
        const blacklist = [
            'id',
            'logDir',
            'startTime',
            'endTime',
            'experimentName',
            'searchSpace',
            'trainingServicePlatform'
        ];
        const filter = (key: string, val: any): any => {
            return blacklist.includes(key) ? undefined : val;
        };
        const profile = lodash.cloneDeep(EXPERIMENT.profile);
        profile.execDuration = convertDuration(profile.execDuration);

        const prettyWidth = innerWidth > 1400 ? 100 : 60;

        return (
            <AppContext.Consumer>
                {(value): React.ReactNode => {
                    const unit = value.maxDurationUnit;
                    profile.params.maxExecDuration = `${convertTimeAsUnit(
                        unit,
                        profile.params.maxExecDuration
                    )}${unit}`;
                    const showProfile = JSON.stringify(profile, filter, 2);
                    return (
                        <Stack>
                            <Panel
                                isOpen={true}
                                hasCloseButton={false}
                                isFooterAtBottom={true}
                                isLightDismiss={true}
                                onLightDismissClick={hideConfigPanel}
                            >
                                <div className='panel'>
                                    {panelName === 'search space' ? (
                                        <div>
                                            <div className='panelName'>Search space</div>
                                            <MonacoEditor
                                                height={monacoEditorHeight}
                                                language='json'
                                                theme='vs-light'
                                                value={prettyStringify(EXPERIMENT.searchSpace, prettyWidth, 2)}
                                                options={MONACO}
                                            />
                                        </div>
                                    ) : (
                                        <div className='profile'>
                                            <div className='panelName'>Config</div>
                                            <MonacoEditor
                                                width='100%'
                                                height={monacoEditorHeight}
                                                language='json'
                                                theme='vs-light'
                                                value={showProfile}
                                                options={MONACO}
                                            />
                                        </div>
                                    )}
                                    <PrimaryButton text='Close' className='configClose' onClick={hideConfigPanel} />
                                </div>
                            </Panel>
                        </Stack>
                    );
                }}
            </AppContext.Consumer>
        );
    }
}

export default TrialConfigPanel;
