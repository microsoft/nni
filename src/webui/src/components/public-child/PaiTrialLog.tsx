import * as React from 'react';
import { Row } from 'antd';
import { DOWNLOAD_IP } from '../../static/const';
import PaiTrialChild from './PaiTrialChild';
import LogPathChild from './LogPathChild';

interface PaitrialLogProps {
    logStr: string;
    id: string;
    logCollection: boolean;
}

class PaitrialLog extends React.Component<PaitrialLogProps, {}> {

    constructor(props: PaitrialLogProps) {
        super(props);

    }

    render() {
        const { logStr, id, logCollection } = this.props;
        const isTwopath = logStr.indexOf(',') !== -1
            ?
            true
            :
            false;
        return (
            <div>
                <div>
                    {
                        isTwopath
                            ?
                            <Row>
                                {
                                    logCollection
                                        ?
                                        <Row>
                                            <a
                                                target="_blank"
                                                href={`${DOWNLOAD_IP}/trial_${id}.log`}
                                                style={{ marginRight: 10 }}
                                            >
                                                Trial stdout
                                            </a>
                                            <a target="_blank" href={logStr.split(',')[1]}>hdfsLog</a>
                                        </Row>
                                        :
                                        <Row>
                                            <LogPathChild
                                                eachLogpath={logStr.split(',')[0]}
                                                logName="Trial stdout:"
                                            />
                                            <LogPathChild
                                                eachLogpath={logStr.split(',')[1]}
                                                logName="Log on HDFS:"
                                            />
                                        </Row>
                                }
                            </Row>
                            :
                            <PaiTrialChild
                                logString={logStr}
                                id={id}
                                logCollect={logCollection}
                            />
                    }
                </div>
            </div>
        );
    }
}

export default PaitrialLog;
