import * as React from 'react';
import { Stack, TooltipHost, getId } from 'office-ui-fabric-react';
import { EXPERIMENT } from '../../static/datamodel';
import { formatTimestamp } from '../../static/function';

interface BasicInfoProps {
    experimentUpdateBroadcast: number;
}

class BasicInfo extends React.Component<BasicInfoProps, {}> {
    // Use getId() to ensure that the ID is unique on the page.
    // (It's also okay to use a plain string without getId() and manually ensure uniqueness.)
    // for tooltip user the log directory
    private _hostId: string = getId('tooltipHost');
    constructor(props: BasicInfoProps) {
        super(props);
    }

    render(): React.ReactNode {
        return (
            <Stack horizontal className="main">
                <Stack.Item grow={100 / 3} className="padItem basic">
                    <p>Name</p>
                    <div>{EXPERIMENT.profile.params.experimentName}</div>
                    <p>ID</p>
                    <div>{EXPERIMENT.profile.id}</div>
                </Stack.Item>
                <Stack.Item grow={100 / 3} className="padItem basic">
                    <p>Start time</p>
                    <div className="nowrap">{formatTimestamp(EXPERIMENT.profile.startTime)}</div>
                    <p>End time</p>
                    <div className="nowrap">{formatTimestamp(EXPERIMENT.profile.endTime)}</div>
                </Stack.Item>
                <Stack.Item grow={100 / 3} className="padItem basic">
                    <p>Log directory</p>
                    <div className="nowrap">
                        {/* <Tooltip placement="top" title={EXPERIMENT.profile.logDir || ''}>
                            {EXPERIMENT.profile.logDir || 'unknown'}
                        </Tooltip> */}
                        {/* need test */}
                        <TooltipHost
                            content={EXPERIMENT.profile.logDir || ''}
                            id={this._hostId}
                            calloutProps={{ gapSpace: 0 }}
                            styles={{ root: { display: 'inline-block' } }}
                        >
                            {EXPERIMENT.profile.logDir || 'unknown'}
                        </TooltipHost>
                    </div>
                    <p>Training platform</p>
                    <div className="nowrap">{EXPERIMENT.profile.params.trainingServicePlatform}</div>
                </Stack.Item>

            </Stack>
        );
    }
}

export default BasicInfo;
