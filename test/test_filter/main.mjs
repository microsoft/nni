import got from 'got'
import simpleGit from 'simple-git'
import multimatch from 'multimatch'

import { parsePullRequstBody } from './parsePullRequestBody.mjs'

const SKIP_VARIABLE = 'skipsubsequent'

const VARIABLES = {
  githubPAT: 'nni.ci.githubPAT',
  markdownHeading: 'nni.ci.markdown.heading',
  markdownOptionIndex: 'nni.ci.markdown.optionIndex',
  markdownOptionValue: 'nni.ci.markdown.optionValue',
  fileChangeGlobs: 'nni.ci.file.globs'
}

const getPipelineVar = (pipelineVar) => {
  const key = pipelineVar.replaceAll('.', '_').toUpperCase()
  return process.env[key]
}

const getGithubPullRequestInfo = async () => {
  const repoId = process.env.BUILD_REPOSITORY_ID
  const prNumber = process.env.SYSTEM_PULLREQUEST_PULLREQUESTNUMBER
  const githubPAT = getPipelineVar(VARIABLES.githubPAT)

  const prUrl = `https://api.github.com/repos/${repoId}/pulls/${prNumber}`
  const opt = {}
  if (typeof (githubPAT) === 'string' && githubPAT.length > 0) {
    const auth = Buffer.from(`:${githubPAT}`).toString('base64')
    opt.headers = {
      Authorization: `Basic ${auth}`
    }
  }
  const resp = await got(prUrl, opt).json()
  return resp
}

const basicCheck = async () => {
  console.log('[Basic Check]')
  const repoProvider = process.env.BUILD_REPOSITORY_PROVIDER
  console.log('Repo Provider:', repoProvider)
  if (repoProvider !== 'GitHub') {
    console.log(`Invalid repo provider ${repoProvider}, run following tests by default.`)
    return true
  }

  const buildReason = process.env.BUILD_REASON
  console.log('Build Reason:', buildReason)
  if (buildReason !== 'PullRequest') {
    console.log('Not triggered by pull request, run following tests by default.')
    return true
  }

  return false
}

const prBodyCheck = async () => {
  console.log('[Pull Request Body]')
  const prInfo = await getGithubPullRequestInfo()
  const prBody = prInfo.body || ''
  const markdownHeading = getPipelineVar(VARIABLES.markdownHeading)
  console.log('Hint Heading:', markdownHeading)
  const selectedOptions = parsePullRequstBody(prBody, markdownHeading)
  console.log('Selected Options:', selectedOptions)

  const optIdx = parseInt(getPipelineVar(VARIABLES.markdownOptionIndex), '10')
  const optValue = getPipelineVar(VARIABLES.markdownOptionValue)
  console.log(`Target Option Index: ${optIdx}`)
  console.log(`Target Option Value: ${optValue}`)

  for (const [value, idx] of selectedOptions) {
    if (value === optValue || idx === optIdx) {
      console.log(`Option ${idx} "${value}" selected, run following tests`)
      return true
    }
  }

  console.log('Option not selected.')
  return false
}

const fileChangeCheck = async () => {
  console.log('[File Changes]')
  const globs = getPipelineVar(VARIABLES.fileChangeGlobs)
  if (typeof (globs) !== 'string' || globs.length === 0) {
    console.log('Glob pattern not provided, skip')
    return false
  }
  const patterns = globs.split(',')
  console.log(`Glob patterns: ${globs}`)

  const targetBranch = process.env.SYSTEM_PULLREQUEST_TARGETBRANCH
  const git = simpleGit()
  const summary = await git.diffSummary(`origin/${targetBranch}`)
  const files = summary.files.map(x => x.file)

  const result = multimatch(files, patterns)
  if (result.length > 0) {
    console.log('Following changes match provided patterns:')
    for (const path of result) {
      console.log(path)
    }
    console.log('Run following tests')
    return true
  }

  console.log('No changes match provided patterns')
  return false
}

const setOutputVariable = (key, value) => {
  console.log(`Set variable: ${key} -> ${value}`)
  console.log(`##vso[task.setvariable variable=${key};isoutput=true]${value}`)
}

const main = async () => {
  const checks = [
    basicCheck,
    fileChangeCheck,
    prBodyCheck
  ]
  for (const check of checks) {
    const res = await check()
    if (res) {
      setOutputVariable(SKIP_VARIABLE, false)
      return
    }
  }

  setOutputVariable(SKIP_VARIABLE, true)
}
main()
